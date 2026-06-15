import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

from generative.loss import KLDiv
from generative.utils_gen import Normalizer, dummy_ctx, get_confounder_dict, compute_dynamic_attraction_loss

class DAFL():
    def __init__(self,args,teacher,student,G,history,W,opt_S,transform):
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
        self.normalizer = Normalizer(args.dataset)

        # Generator Loss terms
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
        
        # Incremental gamma_ood
        self.gamma_adaptive = args.gamma_adaptive
        self.new_gamma_ood = args.new_gamma_ood
        self.old_gamma_ood = args.gamma_ood

        # Generator reset
        self.g_reset = args.g_reset

        # Generator penality
        self.g_penality = args.g_penality
        
        # KD process
        self.kd_steps = args.kd_steps
        self.kl_criterion = KLDiv(T=args.temperature)
        self.autocast = dummy_ctx

        # Student energy matching
        self.s_energy_match = args.s_energy_match
        
        # Process various
        self.log_interval = args.log_interval
        self.device = args.gpu

    def _gamma_tuning(self,epoch):
        if self.gamma_adaptive == "single":
            if epoch > 20:
                self.gamma_ood = self.new_gamma_ood
        elif self.gamma_adaptive == "double":
            if epoch > 230:
                self.gamma_ood = self.old_gamma_ood                    
            elif epoch > 20: 
                self.gamma_ood = self.new_gamma_ood
    
    def _generator_reset(self,epoch):
        if self.g_reset and (epoch % 100 == 0 and epoch > 0):
            print(f"[LOG] Resetting generator output layers at epoch {epoch}")
            conv_layers = [m for m in self.G.conv_blocks.modules() if isinstance(m, nn.Conv2d)]
            
            # self.gamma_ood = 1.0
            if len(conv_layers) >= 2:
                for layer in conv_layers[-2:]:
                    nn.init.normal_(layer.weight, 0.0, 0.02)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    def _diversity_penalty(self,epoch,inputs):
        diversity_loss = -torch.pdist(inputs.view(inputs.size(0), -1)).mean()
        diversity_term = 0.1 * diversity_loss  # before 0.01
        self.history["G_diversity"]["values"].append(diversity_term.item())

        if self.g_penality and epoch > 150:
            return diversity_term
        else:
            return torch.tensor(0.0, device=self.device)

    def _energy_beta(self, epoch):
        if epoch < 50:
            return 0.01
        elif epoch < 100:
            return 0.02
        elif epoch < 150:
            return 0.05
        elif epoch < 200:
            return 0.08
        else:
            return 0.1

    def _energy_matching(self,s_energy,t_energy,epoch):
        energy_diff = s_energy - t_energy
        with torch.no_grad():
            scale = t_energy.abs().mean() + 1e-6
        energy_matching = (energy_diff ** 2).mean() / (scale ** 2)
        t_s_match = self._energy_beta(epoch) * energy_matching
        self.history["T_S_match"]["values"].append(t_s_match.item())

        if self.s_energy_match:    
            return t_s_match
        else:
            return torch.tensor(0.0, device=self.device)

    def update_G(self,epoch):
        # Network setup
        self.student.eval()
        self.G.train()
        self.teacher.eval()

        # Corrective approaches
        self._gamma_tuning(epoch) # Gamma_OOD tuning
        self.history["G_gamma_values"]["values"].append(self.gamma_ood)
        self._generator_reset(epoch) # Generator reset

        # if epoch == 120 or epoch == 220: # g_reset 100
        # if epoch == 70 or epoch == 120 or epoch == 170 or epoch == 220: # g_reset 50
        # if epoch == 30 or epoch == 50 or epoch == 70 or epoch == 90 or epoch == 110 or epoch == 130 or epoch == 150 or epoch == 170 or epoch == 190 or epoch == 210 or epoch == 230: # g_reset 20
            # self.gamma_ood = 0.1
                
        # Generation loop
        gen_bar = tqdm(range(self.iterations), desc="  └── Generation", leave=False)
        for _ in gen_bar:
            self.opt_G.zero_grad()

            # Random sampling
            z = torch.randn(size=(self.synthesis_batch_size, self.nz), device=self.device) # Gaussian noise

            # Generate images
            inputs = self.G(z)
            inputs = self.normalizer(inputs)

            # Teacher output on generated image
            t_out, t_feat = self.teacher(inputs)

            # DAFL base losses
            if not self.ood_loss:
                loss_oh = F.cross_entropy(t_out, t_out.max(1)[1])
                self.history["G_OH_train_loss"]["values"].append(loss_oh.item())

                loss_act = -t_feat.abs().mean()
                self.history["G_ACT_train_loss"]["values"].append(loss_act.item())

                p = F.softmax(t_out, dim=1).mean(0)
                loss_ie = (p * torch.log(p)).sum()
                self.history["G_IE_train_loss"]["values"].append(loss_ie.item())

                loss_energy = torch.tensor(0.0)
                self.history["G_E_loss"]["values"].append(loss_energy.item())

                _ = self._diversity_penalty(epoch,inputs)

                loss_G = loss_oh + self.act * loss_act + self.ie * loss_ie

            # OOD loss
            if self.ood_loss:
                # Energy loss: shape the generated distribution
                energy = -torch.logsumexp(t_out, dim=1)
                loss_energy = energy.mean()
                self.history["G_E_loss"]["values"].append(energy.mean().item())

                p = F.softmax(t_out, dim=1).mean(0)
                loss_ie = (p * torch.log(p)).sum()
                self.history["G_IE_train_loss"]["values"].append(loss_ie.item())

                if self.additive_loss:
                    loss_oh = F.cross_entropy(t_out, t_out.max(1)[1])
                    self.history["G_OH_train_loss"]["values"].append(loss_oh.item())

                    loss_act = -t_feat.abs().mean()
                    self.history["G_ACT_train_loss"]["values"].append(loss_act.item())

                    loss_G = self.gamma_ood * loss_energy + loss_oh + self.act * loss_act + self.ie * loss_ie # ADDITIVE
                
                else:
                    loss_oh = torch.tensor(0.0)
                    self.history["G_OH_train_loss"]["values"].append(loss_oh.item())

                    loss_act = torch.tensor(0.0)
                    self.history["G_ACT_train_loss"]["values"].append(loss_act.item())

                    loss_G = self.gamma_ood * loss_energy + self.ie * loss_ie # STANDALONE
                
                loss_G += self._diversity_penalty(epoch,inputs) # Corrective approaches - Diversity loss

            # KDCI confounders modeling
            if self.kdci:
                self.Z, self.Ps = get_confounder_dict(t_out.detach(), self.confounder_size, pca=False)

            # Stats
            with torch.no_grad():
                teacher_entropy = -(F.softmax(t_out, dim=1) * F.log_softmax(t_out, dim=1)).sum(1).mean()
                self.history["G_T_entropy"]["values"].append(teacher_entropy.item())

            # Backward
            self.history["G_train_loss"].append(loss_G.item())
            loss_G.backward()
            self.opt_G.step()
        
        return loss_G, loss_oh, loss_act, loss_ie, loss_energy

    def update_S(self,epoch):
        # Networks setup
        self.student.train()
        self.teacher.eval()

        # KD loop
        kd_bar = tqdm(range(self.kd_steps), desc="  └── KD", leave=False)
        for _ in kd_bar:

            # Teacher contribute
            with torch.no_grad():
                z = torch.randn(size=(self.sample_batch_size,self.nz), device=self.device)
                images = self.normalizer(self.G(z))
                with self.autocast():
                    t_out, _ = self.teacher(images)
                
                t_energy = -torch.logsumexp(t_out, dim=1)
                self.history["T_S_energy"]["values"].append(t_energy.mean().item())

                teacher_entropy = -(F.softmax(t_out, dim=1) * F.log_softmax(t_out, dim=1)).sum(1).mean()
                self.history["T_S_entropy"]["values"].append(teacher_entropy.item())

            # Student contribute
            s_out = self.student(images.detach())

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

            # Stats
            with torch.no_grad():
                agreement = (s_out.argmax(1) == t_out.argmax(1)).float().mean()
                self.history["T_S_agreement"]["values"].append(agreement.item())

            s_energy = -torch.logsumexp(s_out, dim=1)
            self.history["S_energy"]["values"].append(s_energy.mean().item()  )

            # Student loss
            loss_S = self.kl_criterion(s_out, t_out.detach()) + self._energy_matching(s_energy,t_energy,epoch) # Energy correction            
            
            # Backward
            self.opt_S.zero_grad()
            loss_S.backward()
            self.opt_S.step()
            
            # KD accuracy
            with torch.no_grad():
                s_outputs = s_out.max(1)[1]
                t_outputs = t_out.max(1)[1]
                correct = (s_outputs.view(-1) == t_outputs.view(-1)).sum()
                cnt = torch.numel(t_outputs)
                train_acc = (correct / cnt).detach().cpu()

            # Update history
            self.history["S_train_loss"].append(loss_S.item())
            self.history["S_train_accuracy"].append(train_acc.item())

        return loss_S, train_acc