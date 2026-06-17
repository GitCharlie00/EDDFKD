import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from tqdm import tqdm
from time import time
from kornia import augmentation

from generative.FAST.hooks import DeepInversionHook
from generative.loss import KLDiv, kl_loss, energy_kd_weights, weighted_kl, energy_adaptive_kl
from generative.utils_gen import Normalizer, dummy_ctx, get_confounder_dict
from generative.FAST.fast_utils import ImagePool, DataIter, reset_l0, reptile_grad

class FAST():
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
        self.opt_G = torch.optim.Adam(self.G.parameters(), self.lr_g*self.iterations, betas=[0.5, 0.999])
        self.normalizer = Normalizer(args.dataset)
        self.energy_target = args.energy_target

        # Generator Loss terms
        self.oh = args.oh
        self.adv = args.adv
        self.feat = args.feat

        # FAST elements
        self.ep = 0
        self.warmup = args.warmup
        self.img_size = (3, 32, 32)
        self.num_classes = args.num_classes
        self.lr_z = args.lr_z
        self.data_clear = args.data_clear
        self.keep_last = args.keep_last
        self.data_iter = None
        self.image_pool_dir = save_dir[:-1] + "_pool" + "/"
        os.makedirs(self.image_pool_dir, exist_ok=True)
        self.data_pool = ImagePool(root=self.image_pool_dir)
        self.aug = transforms.Compose([
            augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
            augmentation.RandomHorizontalFlip(),
            self.normalizer,
        ])
        self.bn_mmt = args.bn_mmt
        self.hooks = []
        for m in self.teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append(DeepInversionHook(m, self.bn_mmt))

        # KDCI correction
        self.kdci = args.kdci
        self.confounder_size = args.confounder_size
        self.Tanh = nn.Tanh()

        # Generator energy loss
        self.ood_loss = args.ood_loss
        self.ood_gamma = args.gamma_ood
        self.additive_loss = args.additive_loss
        self.g_reset = args.g_reset

        # KD process
        self.kd_steps = args.kd_steps
        self.temperature = args.temperature
        self.kl_criterion = KLDiv(T=args.temperature)
        self.autocast = dummy_ctx

        # Energy-weighted KD (reliability reweighting at distillation time)
        self.energy_kd = args.energy_kd
        self.energy_kd_beta = args.energy_kd_beta

        # Energy-adaptive distillation temperature
        self.energy_temp = args.energy_temp
        self.energy_temp_base = args.energy_temp_base
        self.energy_temp_alpha = args.energy_temp_alpha

        # Process various
        self.log_interval = args.log_interval
        self.device = args.gpu

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

    def _energy_matching(self,s_energy,t_energy):
        energy_diff = s_energy - t_energy
        scale = t_energy.abs().mean() + 1e-6
        energy_matching = (energy_diff ** 2).mean() / (scale ** 2)
        t_s_match = 0.05 * energy_matching
        self.history["T_S_match"]["values"].append(t_s_match.item())

    def update_G(self,epoch):
        # Network setup
        self.ep += 1
        self.student.eval()
        self.teacher.eval()
        self.G.train()
        best_cost = 1e6

        # Random sampling
        best_inputs = None
        z = torch.randn(size=(self.synthesis_batch_size, self.nz), device=self.device).requires_grad_()
        targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,))
        targets = targets.to(self.device)
        
        fast_G = self.G.clone(self.device)

        opt_fast_G = optim.Adam([
            {'params': fast_G.parameters()},
            {'params': [z], 'lr': self.lr_z}
        ], lr=self.lr_g, betas=[0.5, 0.999])

        # Generation loop
        gen_bar = tqdm(range(self.iterations), desc="  └── Generation", leave=False)
        for _ in gen_bar:
            # Generate images
            inputs = fast_G(z)
            inputs_aug = self.aug(inputs)

            # Teacher output on generated image
            t_out, t_feat = self.teacher(inputs_aug)

            # FAST base losses
            if not self.ood_loss:
                loss_oh = F.cross_entropy(t_out, targets)
                self.history["G_OH_train_loss"]["values"].append(loss_oh.item())

                if self.ep >= self.warmup:
                    s_out = self.student(inputs_aug)
                    mask = (s_out.max(1)[1] == t_out.max(1)[1]).float()
                    loss_adv = -(kl_loss(s_out, t_out, reduction='none').sum(1) * mask).mean()  # decision adversarial distillation
                    self.history["G_ADV_train_loss"]["values"].append(loss_adv.item())
                else:
                    loss_adv = loss_oh.new_zeros(1)
                    self.history["G_ADV_train_loss"]["values"].append(loss_adv.item())

                loss_feat = sum([h.r_feature for h in self.hooks])
                self.history["G_FEAT_train_loss"]["values"].append(loss_feat.item())

                loss_energy = torch.tensor(0.0)
                self.history["G_E_loss"]["values"].append(loss_energy.item())

                loss_G = self.oh * loss_oh + self.adv * loss_adv + self.feat * loss_feat

                gen_bar.set_postfix({
                    "L_oh":loss_oh.item(),
                    "L_adv":loss_adv.item(),
                    "L_feat":loss_feat.item(),
                    "L_g":loss_G.item()
                    })
            # OOD loss
            if self.ood_loss:
                # Generator reset
                # self._generator_reset(epoch)

                # Energy loss: shape the generated distribution
                energy = -torch.logsumexp(t_out, dim=1)
                # loss_energy = energy.mean()
                loss_energy = (energy.mean() - self.energy_target).abs()
                self.history["G_E_loss"]["values"].append(energy.mean().item())

                loss_feat = sum([h.r_feature for h in self.hooks])
                self.history["G_FEAT_train_loss"]["values"].append(loss_feat.item())

                # Additive
                if self.additive_loss:
                    loss_oh = F.cross_entropy(t_out, targets)
                    self.history["G_OH_train_loss"]["values"].append(loss_oh.item())

                    if self.ep >= self.warmup:
                        s_out = self.student(inputs_aug)
                        mask = (s_out.max(1)[1] == t_out.max(1)[1]).float()
                        loss_adv = -(kl_loss(s_out, t_out, reduction='none').sum(1) * mask).mean()  # decision adversarial distillation
                        self.history["G_ADV_train_loss"]["values"].append(loss_adv.item())
                    else:
                        loss_adv = loss_oh.new_zeros(1)
                        self.history["G_ADV_train_loss"]["values"].append(loss_adv.item())
                    
                    current_gamma = self.ood_gamma if epoch >= 50 else 0.0
                    loss_G = current_gamma * loss_energy + self.oh * loss_oh + self.adv * loss_adv + self.feat * loss_feat
                
                # Substitutive
                else:
                    loss_oh = torch.tensor(0.0)
                    self.history["G_OH_train_loss"]["values"].append(loss_oh.item())

                    loss_adv = torch.tensor(0.0)
                    self.history["G_ADV_train_loss"]["values"].append(loss_adv.item())

                    loss_G = self.ood_gamma * loss_energy + self.feat * loss_feat

            # KDCI confounders modeling
            if self.kdci:
                self.Z, self.Ps = get_confounder_dict(t_out.detach(), self.confounder_size, pca=False)

            # Statistics
            with torch.no_grad():
                self.history["G_lr_values"]["values"].append(self.opt_S.param_groups[0]['lr'])
                teacher_entropy = -(F.softmax(t_out, dim=1) * F.log_softmax(t_out, dim=1)).sum(1).mean()
                self.history["G_T_entropy"]["values"].append(teacher_entropy.item())
                self._diversity_penalty(epoch,inputs)

            with torch.no_grad():
                if best_cost > loss_G.item() or best_inputs is None:
                    best_cost = loss_G.item()
                    best_inputs = inputs.data
            
            # Backward - inner
            opt_fast_G.zero_grad()
            loss_G.backward()
            opt_fast_G.step()
            
            self.history["G_train_loss"].append(loss_G.item())

        # Backward - outer
        self.opt_G.zero_grad()
        reptile_grad(self.G,fast_G,self.device)
        self.opt_G.step()

        if self.bn_mmt != 0:
            for h in self.hooks:
                h.update_mmt()

        # Synthetic data aggregation
        self.data_pool.add(best_inputs)
        dst = self.data_pool.get_dataset(transform=self.transform)
        loader = torch.utils.data.DataLoader(dst, batch_size=self.sample_batch_size, shuffle=True, num_workers=4, pin_memory=True, sampler=None)
        self.data_iter = DataIter(loader)

        return loss_G, loss_oh, loss_adv, loss_feat, loss_energy

    def update_S(self,epoch):
        # Networks setup
        self.student.train()
        self.teacher.eval()

        # KD loop
        kd_bar = tqdm(range(self.kd_steps), desc="  └── KD", leave=False)
        for _ in kd_bar:
            # Sampling synthetic batch
            images = self.data_iter.next()
            images = images.cuda(self.device, non_blocking=True)

            # Teacher contribute
            with self.autocast():
                with torch.no_grad():
                    # Teacher prediction on synthetic batch
                    t_out, t_feat = self.teacher(images)

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
            if self.energy_temp:
                loss_S = energy_adaptive_kl(s_out, t_out.detach(),
                                            tau_base=self.energy_temp_base,
                                            alpha=self.energy_temp_alpha)
            elif self.energy_kd:
                w = energy_kd_weights(t_out.detach(), beta=self.energy_kd_beta, T=self.temperature)
                loss_S = weighted_kl(s_out, t_out.detach(), weights=w, T=self.temperature)
            else:
                loss_S = self.kl_criterion(s_out, t_out.detach())
            self.opt_S.zero_grad()

            # Backward
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

            kd_bar.set_postfix({
                "Loss":loss_S.item(),
                "Acc":train_acc.item(),
                "lr":self.opt_S.param_groups[0]['lr']
            })

        return loss_S, train_acc