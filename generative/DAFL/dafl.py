import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

from generative.loss import dafl_kdloss, energy_kd_weights, weighted_kl
from generative.utils_gen import get_confounder_dict

class DAFL():
    def __init__(self,args,teacher,student,G,history,W,opt_S,transform,save_dir):
        # Passed from main
        self.args = args
        self.teacher = teacher
        self.student = student
        self.G = G
        self.history = history
        self.Wq = W[0].to(args.gpu) if args.kdci else None
        self.Wk = W[1].to(args.gpu) if args.kdci else None
        self.Wt = W[2].to(args.gpu) if args.kdci else None
        self.opt_S = opt_S
        self.transform = transform

        # Generative process
        self.iterations = args.g_steps
        self.lr_g = args.lr_G
        self.nz = args.z_dim
        self.synthesis_batch_size = args.train_batch_size
        self.sample_batch_size = args.train_batch_size
        self.opt_G = optim.Adam(G.parameters(), lr=self.lr_g)

        # Generator Loss terms
        self.oh = args.oh
        self.act = args.act
        self.ie = args.ie
        
        # KDCI correction
        self.kdci = args.kdci
        self.confounder_size = args.confounder_size
        self.Tanh = nn.Tanh()
        
        # Generator energy loss
        self.ood_loss = args.ood_loss
        self.gamma_ood = args.gamma_ood
        self.additive_loss = args.additive_loss        
        # self.energy_target = args.energy_target
        self.in_energy_target = args.in_energy_target
        self.out_energy_target = args.out_energy_target

        # KD process
        self.kd_steps = args.kd_steps
        self.kl_criterion = dafl_kdloss

        # Energy-weighted KD (reliability reweighting at distillation time)
        self.energy_kd = args.energy_kd
        self.energy_kd_beta = args.energy_kd_beta
        
        # Process various
        self.log_interval = args.log_interval
        self.device = args.gpu

    # def _generator_reset(self,epoch):
    #     if self.g_reset and (epoch % 100 == 0 and epoch > 0):
    #         print(f"[LOG] Resetting generator output layers at epoch {epoch}")
    #         conv_layers = [m for m in self.G.conv_blocks.modules() if isinstance(m, nn.Conv2d)]
            
    #         # self.gamma_ood = 1.0
    #         if len(conv_layers) >= 2:
    #             for layer in conv_layers[-2:]:
    #                 nn.init.normal_(layer.weight, 0.0, 0.02)
    #                 if layer.bias is not None:
    #                     nn.init.constant_(layer.bias, 0)

    def _generator_reset(self, epoch):
        # if self.g_reset and (epoch % 100 == 0 and epoch > 0):
        if self.g_reset and (epoch % 250 == 0 and epoch > 0) and epoch < 1800:
            print(f"[LOG] Resetting generator output layers at epoch {epoch}")
            conv_layers = [m for m in self.G.modules() if isinstance(m, nn.Conv2d)]

            if len(conv_layers) >= 2:
                for layer in conv_layers[-2:]:
                    nn.init.normal_(layer.weight, 0.0, 0.02)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    def _diversity_penalty(self,epoch,inputs):
        diversity_loss = -torch.pdist(inputs.view(inputs.size(0), -1)).mean()
        diversity_term = 0.1 * diversity_loss  # before 0.01
        self.history["G_diversity"]["values"].append(diversity_term.item())

    def _energy_matching(self,s_energy,t_energy):
        energy_diff = s_energy - t_energy
        scale = t_energy.abs().mean() + 1e-6
        energy_matching = (energy_diff ** 2).mean() / (scale ** 2)
        t_s_match = 0.05 * energy_matching
        self.history["T_S_match"]["values"].append(t_s_match.item())

    def dafl_loop(self,epoch):
        # --- Update G --- #

        # Network setup
        self.student.train()
        self.G.train()
        self.teacher.eval()

        # Generation loop
        gen_bar = tqdm(range(self.iterations), desc="  └── Generation", leave=False)
        for _ in gen_bar:
            # Random sampling
            self.opt_G.zero_grad()
            z = torch.randn(size=(self.synthesis_batch_size, self.nz), device=self.device) # Gaussian noise

            # Generate images
            gen_images = self.G(z)

            # Teacher output on generated image
            t_out, t_feat = self.teacher(gen_images)

            # DAFL base losses
            if not self.ood_loss:
                loss_oh = F.cross_entropy(t_out, t_out.max(1)[1])
                self.history["G_OH_train_loss"]["values"].append(loss_oh.item())

                loss_act = -t_feat.abs().mean()
                self.history["G_ACT_train_loss"]["values"].append(loss_act.item())

                p = F.softmax(t_out, dim=1).mean(0)
                loss_ie = (p * torch.log10(p)).sum()
                self.history["G_IE_train_loss"]["values"].append(loss_ie.item())

                loss_energy = torch.tensor(0.0)
                self.history["G_E_loss"]["values"].append(loss_energy.item())

                loss_G = self.oh * loss_oh + self.act * loss_act + self.ie * loss_ie
                self.history["G_train_loss"].append(loss_G.item())
            
            # OOD loss
            if self.ood_loss:
                # # ------- OLD 
                # # Energy loss: shape the generated distribution
                # energy = -torch.logsumexp(t_out, dim=1)
                # # loss_energy = energy.mean()
                # #  ---
                # loss_energy = (energy.mean() - self.energy_target).abs()
                # #  ---
                # # loss_energy = (energy.mean() - self.energy_target) ** 2
                # #  --- 
                # # margin = 2.0
                # # loss_energy = torch.relu(energy - self.energy_target + margin).mean()
                # # min_energy = -11.0  # Lower bound
                # # max_energy = -8.0   # Upper bound
                # # loss_too_high = torch.relu(energy - max_energy).mean()
                # # loss_too_low = torch.relu(min_energy - energy).mean()
                # # loss_energy = loss_too_high + loss_too_low
                
                # # ------- NEW
                # Calcola l'energia
                energy = -torch.logsumexp(t_out, dim=1)
                
                # Margini ispirati al paper (da passare possibilmente via args)
                m_in = self.in_energy_target   # Limite inferiore (energia tipica dei dati in-distribution)
                m_out = self.out_energy_target  # Limite superiore (energia tipica dei dati OOD)
                
                # 1. Penalizza i campioni con energia troppo alta (troppo OOD)
                # Equivalente a: max(0, energy - m_out)^2
                loss_high = torch.relu(energy - m_out).pow(2).mean()
                
                # 2. Penalizza i campioni con energia troppo bassa (eccessiva confidenza/overfitting)
                # Equivalente a: max(0, m_in - energy)^2
                loss_low = torch.relu(m_in - energy).pow(2).mean()
                
                # La loss finale è la somma delle due penalità
                loss_energy = loss_high + loss_low
                
                self.history["G_E_loss"]["values"].append(loss_energy.item())

                # self.history["G_E_loss"]["values"].append(energy.mean().item())
                self.history["G_E_loss"]["values"].append(loss_energy.item())

                p = F.softmax(t_out, dim=1).mean(0)
                loss_ie = (p * torch.log10(p)).sum()
                self.history["G_IE_train_loss"]["values"].append(loss_ie.item())

                ## ------- OLD
                # # Additive
                # if self.additive_loss:
                #     loss_oh = F.cross_entropy(t_out, t_out.max(1)[1])
                #     self.history["G_OH_train_loss"]["values"].append(loss_oh.item())

                #     loss_act = -t_feat.abs().mean()
                #     self.history["G_ACT_train_loss"]["values"].append(loss_act.item())

                #     current_gamma = self.gamma_ood if epoch >= 500 else 0.0
                    
                #     loss_G = current_gamma * loss_energy + self.oh * loss_oh + self.act * loss_act + self.ie * loss_ie
                #     self.history["G_train_loss"].append(loss_G.item())
                
                # # Substitutive
                # else:
                #     loss_oh = torch.tensor(0.0)
                #     self.history["G_OH_train_loss"]["values"].append(loss_oh.item())

                #     loss_act = torch.tensor(0.0)
                #     self.history["G_ACT_train_loss"]["values"].append(loss_act.item())

                #     loss_G = self.gamma_ood * loss_energy + self.ie * loss_ie
                #     self.history["G_train_loss"].append(loss_G.item())

                loss_oh = F.cross_entropy(t_out, t_out.max(1)[1])
                self.history["G_OH_train_loss"]["values"].append(loss_oh.item())

                loss_act = -t_feat.abs().mean()
                self.history["G_ACT_train_loss"]["values"].append(loss_act.item())

                current_gamma = self.gamma_ood if epoch >= 500 else 0.0
                
                loss_G = current_gamma * loss_energy + self.oh * loss_oh + self.act * loss_act + self.ie * loss_ie
                self.history["G_train_loss"].append(loss_G.item())
                
            # KDCI confounders modeling
            if self.kdci:
                self.Z, self.Ps = get_confounder_dict(t_out.detach(), self.confounder_size, pca=False)

            # Statistics
            with torch.no_grad():
                self.history["G_lr_values"]["values"].append(self.opt_S.param_groups[0]['lr'])
                teacher_entropy = -(F.softmax(t_out, dim=1) * F.log_softmax(t_out, dim=1)).sum(1).mean()
                self.history["G_T_entropy"]["values"].append(teacher_entropy.item())
                self._diversity_penalty(epoch,gen_images)

        # --- Update S --- #
        
        # Networks setup
        # self.student.train()
        self.teacher.eval()

        # KD loop
        kd_bar = tqdm(range(self.kd_steps), desc="  └── KD", leave=False)
        for _ in kd_bar:
            # Student contribute
            self.opt_S.zero_grad()
            s_out = self.student(gen_images.detach())

            # KDCI correction
            if self.kdci:
                query = self.Wq(s_out)
                query_expand = query.unsqueeze(1)
                key =  self.Wk(self.Z)
                
                fuse = query_expand + key
                fuse = self.Tanh(fuse)

                attention = self.Wt(fuse)
                attention = F.softmax(attention, dim=1)

                proportions = self.Ps.unsqueeze(1)
                values = self.Z * proportions
                lambda_squeezed = attention.squeeze(-1)
                F_z = torch.matmul(lambda_squeezed, values)

                s_out = s_out + F_z

            # Statistics
            with torch.no_grad():
                t_energy = -torch.logsumexp(t_out, dim=1)
                self.history["T_S_energy"]["values"].append(t_energy.mean().item())

                teacher_entropy = -(F.softmax(t_out, dim=1) * F.log_softmax(t_out, dim=1)).sum(1).mean()
                self.history["T_S_entropy"]["values"].append(teacher_entropy.item())

                agreement = (s_out.argmax(1) == t_out.argmax(1)).float().mean()
                self.history["T_S_agreement"]["values"].append(agreement.item())

                s_energy = -torch.logsumexp(s_out, dim=1)
                self.history["S_energy"]["values"].append(s_energy.mean().item())

                self._energy_matching(s_energy,t_energy)

            # Student loss
            if self.energy_kd:
                w = energy_kd_weights(t_out.detach(), beta=self.energy_kd_beta)
                loss_S = weighted_kl(s_out, t_out.detach(), weights=w)
            else:
                loss_S = self.kl_criterion(s_out, t_out.detach())
            self.history["S_train_loss"].append(loss_S.item())
        
        # KD current accuracy
        with torch.no_grad():
            s_outputs = s_out.max(1)[1]
            t_outputs = t_out.max(1)[1]
            correct = (s_outputs.view(-1) == t_outputs.view(-1)).sum()
            cnt = torch.numel(t_outputs)
            train_acc = (correct / cnt).detach().cpu()
            self.history["S_train_accuracy"].append(train_acc.item())
        
        # Backward pass
        loss = loss_G + loss_S
        loss.backward()
        self.opt_G.step()
        self.opt_S.step()

        # print(f"\n[DEBUG KD Loss]")
        # print(f"  one_hot: {self.oh:.6f}")
        # print(f"  l_one_hot: {loss_oh.item():.6f}")
        # print(f"  ie: {self.ie:.6f}")
        # print(f"  l_ie: {loss_ie.item():.6f}")
        # print(f"  a: {self.act:.6f}")
        # print(f"  l_a: {loss_act.item():.6f}")
        # print(f"  L_g generator loss: {loss_G.item():.6f}")
        # print(f"  s_out shape: {s_out.shape}")
        # print(f"  t_out shape: {t_out.shape}")
        # print(f"  s_out min/max: {s_out.min():.4f} / {s_out.max():.4f}")
        # print(f"  t_out min/max: {t_out.min():.4f} / {t_out.max():.4f}")
        
        # Calcola manualmente come official
        # p = F.log_softmax(s_out, dim=1)
        # q = F.softmax(t_out.detach(), dim=1)
        # manual_kl = F.kl_div(p, q, reduction='sum') / s_out.shape[0]
        # print(f"  Manual KL (official style): {manual_kl.item():.6f}")
        # print(f"  Loss KL (batchmean): {loss_S.item():.6f}")
        # print(f"L_G+L_s total backpropagated loss: {loss.item():.6f}")
        # exit(0)

        return [loss_G, loss_oh, loss_act, loss_ie, loss_energy], [loss_S, train_acc]