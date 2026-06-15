import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from tqdm import tqdm

from generative.DEEPINV.hooks import DeepInversionHook
from generative.loss import KLDiv, kl_loss
from generative.utils_gen import Normalizer, dummy_ctx, get_confounder_dict
from generative.DEEPINV.deepinv_utils import ImagePool, DataIter, jitter_and_flip, get_image_prior_losses, clip_images, jsdiv

class DEEPINV():
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

        # From args
        self.save_dir = args.save_path
        self.iterations = args.g_steps
        self.lr_g = args.lr_G
        self.nz = args.z_dim
        self.adv = args.adv
        self.bn = args.bn
        self.oh = args.oh
        self.num_classes = args.num_classes
        self.synthesis_batch_size = args.train_batch_size
        self.sample_batch_size = args.train_batch_size
        self.bn_mmt = args.bn_mmt
        self.lr_z = args.lr_z
        self.data_clear = args.data_clear
        self.keep_last = args.keep_last
        self.beta_1 = args.beta_1
        self.beta_2 = args.beta_2
        self.warmup = args.warmup
        self.kd_steps = args.kd_steps
        self.log_interval = args.log_interval
        self.ood_loss = args.ood_loss
        self.kdci = args.kdci
        self.confounder_size = args.confounder_size
        self.attraction_scale = args.attraction_scale
        self.device = args.gpu
        
        # Common among methods
        self.kl_criterion = KLDiv(T=args.temperature)
        self.normalizer = Normalizer(args.dataset)
        self.autocast = dummy_ctx

        # Method specific
        self.ood_gamma = 0.1
        self.m_in = -9.0
        self.lambda_ent = 0.1
        self.image_pool_dir = (args.save_path + "/" + args.method + "/" + args.dataset + "/" + args.t_network + "_" + args.s_network + "/").lower()
        self.data_pool = ImagePool(root=self.image_pool_dir)
        self.data_iter = None
        self.img_size = (3, 32, 32)
        self.progressive_scale = False
        self.adv = args.adv
        self.bn = args.bn
        self.oh = args.oh
        self.tv = args.tv
        self.l2 = 0.0

        self.hooks = []
        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append(DeepInversionHook(m, 0))
        self.s_hooks = []
        for m in student.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.s_hooks.append(DeepInversionHook(m, 0))
        assert len(self.hooks) > 0, 'input model should contains at least one BN layer for DeepInversion'

    def update_G(self,epoch):
        self.student.eval()
        best_cost = 1e6
        inputs = torch.randn(size=[self.synthesis_batch_size, *self.img_size], device=self.device).requires_grad_()

        targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,))
        targets = targets.sort()[0]
        targets = targets.to(self.device)

        optimizer = torch.optim.Adam([inputs], self.lr_g, betas=[self.beta_1, self.beta_2])

        best_inputs = inputs.data
        gen_bar = tqdm(range(self.iterations), desc="  └── Generation", leave=False)
        for it in gen_bar:
            inputs_aug = jitter_and_flip(inputs)
            t_out, t_feat = self.teacher(inputs_aug)
            if self.adv > 0:
                s_out = self.student(inputs_aug)
                loss_adv = -jsdiv(s_out, t_out, T=3)
                self.history["G_ADV_train_loss"]["values"].append(loss_adv.item())
            else:
                loss_adv = loss_oh.new_zeros(1)
                self.history["G_ADV_train_loss"]["values"].append(loss_adv.item())

            loss_bn = sum([h.r_feature for h in self.hooks]) - 0.001 * sum([h.r_feature for h in self.s_hooks])
            self.history["G_BN_train_loss"]["values"].append(loss_bn.item())
            loss_oh = F.cross_entropy(t_out, targets)
            self.history["G_OH_train_loss"]["values"].append(loss_oh.item())

            loss_tv = get_image_prior_losses(inputs)
            self.history["G_TV_train_loss"]["values"].append(loss_tv.item())
            loss_G = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv + self.tv * loss_tv

            if self.ood_loss:
                if epoch > 200:
                    # Entropy loss: push the cluster togheter
                    self.Z, self.Ps = get_confounder_dict(t_feat, self.confounder_size, pca=False)
                    loss_entropy = compute_dynamic_attraction_loss(t_out, t_feat, self.Z, self.Ps, self.attraction_scale)
                    self.history["G_ENT_loss"]["values"].append(loss_entropy.item())

                    # Energy loss: shape the generated distribution
                    energy = -torch.logsumexp(t_out, dim=1)
                    self.history["G_ENV_val"]["values"].append(energy.mean().item())
                    penalty = F.relu(energy - self.m_in)
                    loss_energy = penalty.pow(2).mean()
                    self.history["G_ENL_loss"]["values"].append(loss_energy.item())

                    loss_G += self.lambda_ent * loss_entropy + self.ood_gamma * loss_energy            
                else:
                    self.history["G_ENT_loss"]["values"].append(0.0)
                    self.history["G_ENV_val"]["values"].append(0.0)
                    self.history["G_ENL_loss"]["values"].append(0.0)
            
            if self.kdci:
                self.Z, self.Ps = get_confounder_dict(t_out, self.confounder_size)

            if best_cost > loss_G.item():
                best_cost = loss_G.item()
                best_inputs = inputs.data

            optimizer.zero_grad()
            loss_G.backward()
            optimizer.step()
            inputs.data = clip_images(inputs.data, self.normalizer.mean, self.normalizer.std)

            self.history["G_train_loss"].append(loss_G.item())
        
        self.student.train()

        if self.normalizer:
            best_inputs = self.normalizer(best_inputs, True)

        self.data_pool.add(best_inputs)
        if (epoch + 1) % self.data_clear == 0:
            self.data_pool.clear_old_images(keep_last_n=self.keep_last)
            print(f"[LOG] Pruned image pool, kept last {self.keep_last} images")

        dst = self.data_pool.get_dataset(transform=self.transform)
        
        train_sampler = None
        loader = torch.utils.data.DataLoader(
            dst, batch_size=self.sample_batch_size, shuffle=(train_sampler is None),
            num_workers=4, pin_memory=True, sampler=train_sampler)
        self.data_iter = DataIter(loader)

        if self.ood_loss:
            if epoch > 200:
                return_values = [[loss_G, loss_oh, loss_bn, loss_adv, loss_tv], [loss_entropy, energy.mean(), loss_energy, torch.tensor(self.m_in), torch.tensor(self.ood_gamma)]] 
            else:
                return_values = [[loss_G, loss_oh, loss_bn, loss_adv, loss_tv], [torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)]]
        else:
            return_values = [[loss_G, loss_oh, loss_bn, loss_adv, loss_tv], []]
        
        return return_values

    def update_S(self):
        self.student.train()
        self.teacher.eval()

        kd_bar = tqdm(range(self.kd_steps), desc="  └── KD", leave=False)
        for i in kd_bar:
            with torch.no_grad():
                images = self.data_iter.next()
                images = images.cuda(self.device, non_blocking=True)
                
                with self.autocast():
                    t_out, _ = self.teacher(images)

            s_out = self.student(images.detach())

            if self.kdci:
                attn_scores = []
                for i in range(self.Z.shape[0]):  # per ogni cluster
                    score = self.Wt(torch.tanh(self.Wq(s_out) + self.Wk(self.Z[i])))
                    attn_scores.append(score)    
                attn_scores = torch.stack(attn_scores, dim=1)  # [batch, N] Accumula gli score in una variabile
                
                lambda_i = torch.softmax(attn_scores, dim=1)  # [batch, N] Applica softmax per ottenere i pesi
                lambda_i = lambda_i.squeeze(-1)  # Fai attenzione a questo passaggio, non farlo se la forma è già [batch, N] Evita il .squeeze() se la forma è già corretta

                weighted_z = self.Z * self.Ps.view(-1, 1)  # [N, d] # Pondera Z con le proporzioni Ps
                F = torch.einsum('bn,nd->bd', lambda_i, weighted_z)  # [batch, d] Esegui l'operazione einsum per combinare i pesi e i centroidi

                s_out = s_out + F.detach()  # Compensazione della predizione dello studente

            loss_S = self.kl_criterion(s_out, t_out.detach())
            self.opt_S.zero_grad()

            loss_S.backward()
            self.opt_S.step()
            
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
    
    def get_data_pool(self):
        return self.data_pool