import torch
import torch.nn as nn
import torchvision.transforms as transforms

from tqdm import tqdm

from generative.CMI.cmi_utils import *
from generative.CMI.hooks import InstanceMeanHook, DeepInversionHook
from generative.loss import KLDiv, kl_loss
from generative.utils_gen import Normalizer, dummy_ctx, get_confounder_dict

class CMI():
    def __init__(self,args,teacher,student,G,history,W,opt_S,transform):
        # Passed from main
        self.args = args
        self.teacher = teacher
        self.student = student
        self.G = G.train()
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
        self.cr = args.cr
        self.cr_T = args.cr_T
        self.data_clear = args.data_clear
        self.keep_last = args.keep_last
        self.beta_1 = args.beta_1
        self.beta_2 = args.beta_2
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
        self.img_size = (3, 32, 32)    
        self.progressive_scale = False
        self.n_neg = 4096
        self.bank_size = 40960
        self.init_dataset = None
        self.image_pool_dir = (args.save_path + "/" + args.method + "/" + args.dataset + "/" + args.t_network + "_" + args.s_network + "/").lower()
        self.data_pool = ImagePool(root=self.image_pool_dir)
        self.data_iter = None

        self.cmi_hooks = []
        self.feature_layers = None
        if self.args.t_network == "resnet-34":
            self.feature_layers = [self.teacher.layer1, self.teacher.layer2, self.teacher.layer3, self.teacher.layer4]

        if self.feature_layers is not None:
            for layer in self.feature_layers:
                self.cmi_hooks.append(InstanceMeanHook(layer))
        else:
            for m in teacher.modules():
                if isinstance(m, nn.BatchNorm2d):
                    self.cmi_hooks.append(InstanceMeanHook(m))

        with torch.no_grad():
            self.teacher.eval()
            fake_inputs = torch.randn(size=(1, *self.img_size), device=self.device)
            _ = self.teacher(fake_inputs)
            cmi_feature = torch.cat([h.instance_mean for h in self.cmi_hooks], dim=1)
            del fake_inputs

        self.mem_bank = MemoryBank("cpu", max_size=self.bank_size, dim_feat=2*cmi_feature.shape[1])

        self.head = MLPHead(cmi_feature.shape[1], 256).to(self.device).train()
        self.optimizer_head = torch.optim.Adam(self.head.parameters(), lr=args.lr_G)

        self.hooks = []
        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append(DeepInversionHook(m, 0))

        self.aug = MultiTransform([
            transforms.Compose([
                transforms.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
                transforms.RandomHorizontalFlip(),
                self.normalizer,
            ]),
            transforms.Compose([
                transforms.RandomResizedCrop(size=[self.img_size[-2], self.img_size[-1]], scale=[0.25, 1.0]),
                transforms.RandomHorizontalFlip(),
                self.normalizer,
            ]),
        ])
        self.feature_reuse = False

    def update_G(self,epoch):
        self.student.eval()
        self.teacher.eval()
        best_cost = 1e6

        best_inputs = None
        z = torch.randn(size=(self.synthesis_batch_size, self.nz), device=self.device).requires_grad_()
        targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,))
        targets = targets.sort()[0]
        targets = targets.to(self.device)

        if not self.feature_reuse:
            reset_model(self.G)
        opt_G = torch.optim.Adam(
            [{'params': self.G.parameters()}, {'params': [z]}], 
            self.lr_g, betas=[self.beta_1, self.beta_2]
        )

        gen_bar = tqdm(range(self.iterations), desc="  └── Generation", leave=False)
        for it in gen_bar:
            inputs = self.G(z)
            global_view, local_view = self.aug(inputs)

            t_out, _ = self.teacher(global_view)
            loss_bn = sum([h.r_feature for h in self.hooks])
            self.history["G_BN_train_loss"]["values"].append(loss_bn.item())
            
            loss_oh = F.cross_entropy( t_out, targets )
            self.history["G_OH_train_loss"]["values"].append(loss_oh.item())

            if self.adv>0:
                s_out = self.student(global_view)
                mask = (s_out.max(1)[1]==t_out.max(1)[1]).float()
                loss_adv = -(kl_loss(s_out, t_out, reduction='none').sum(1) * mask).mean()
                self.history["G_ADV_train_loss"]["values"].append(loss_adv.item())
            else:
                loss_adv = loss_oh.new_zeros(1)
                self.history["G_ADV_train_loss"]["values"].append(loss_adv.item())

            loss_inv = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv

            if self.cr>0:
                global_feature = torch.cat([ h.instance_mean for h in self.cmi_hooks ], dim=1)
                _ = self.teacher(local_view)
                local_feature = torch.cat([ h.instance_mean for h in self.cmi_hooks ], dim=1)
                cached_feature, _ = self.mem_bank.get_data(self.n_neg)
                cached_local_feature, cached_global_feature = torch.chunk(cached_feature.to(self.device), chunks=2, dim=1)
                proj_feature = self.head( torch.cat([local_feature, cached_local_feature, global_feature, cached_global_feature], dim=0) )
                proj_local_feature, proj_global_feature = torch.chunk(proj_feature, chunks=2, dim=0)
                cr_logits = torch.mm(proj_local_feature, proj_global_feature.detach().T) / self.cr_T # (N + N') x (N + N')
                cr_labels = torch.arange(start=0, end=len(cr_logits), device=self.device)
                loss_cr = F.cross_entropy( cr_logits, cr_labels, reduction='none')  #(N + N')
                if self.mem_bank.n_updates>0:
                    loss_cr = loss_cr[:self.synthesis_batch_size].mean() + loss_cr[self.synthesis_batch_size:].mean()
                    self.history["G_CR_train_loss"]["values"].append(loss_cr.item())
                else:
                    loss_cr = loss_cr.mean()
                    self.history["G_CR_train_loss"]["values"].append(loss_cr.item())
            else: 
                loss_cr = loss_inv.new_zeros(1)
                self.history["G_CR_train_loss"]["values"].append(loss_cr.item())

            loss = self.cr * loss_cr + loss_inv
            with torch.no_grad():
                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()
                    best_inputs = inputs.data
                    best_features = torch.cat([local_feature.data, global_feature.data], dim=1).data
            

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
                z = torch.randn(size=(self.synthesis_batch_size, self.nz), device=self.device)
                X = self.G(z)
                X = self.normalizer(X)

                M, _ = self.teacher(X)
                self.Z, self.Ps = get_confounder_dict(M,self.confounder_size)     

            opt_G.zero_grad()
            self.optimizer_head.zero_grad()
            self.history["G_train_loss"].append(loss.item())
            loss.backward()
            opt_G.step()
            self.optimizer_head.step()

        self.student.train()
        self.data_pool.add(best_inputs)
        if (epoch + 1) % self.data_clear == 0:
            self.data_pool.clear_old_images(keep_last_n=self.keep_last)
            print(f"[LOG] Pruned image pool, kept last {self.keep_last} images")
        self.mem_bank.add(best_features)

        dst = self.data_pool.get_dataset(transform=self.transform)
        train_sampler = None
        loader = torch.utils.data.DataLoader(
            dst, batch_size=self.sample_batch_size, shuffle=(train_sampler is None),
            num_workers=4, pin_memory=True, sampler=train_sampler)
        self.data_iter = DataIter(loader)

        if self.ood_loss:
            if epoch > 200:
                return_values = [[loss, loss_bn, loss_oh, loss_adv, loss_cr], [loss_entropy, energy.mean(), loss_energy, torch.tensor(self.m_in), torch.tensor(self.ood_gamma)]] 
            else:
                return_values = [[loss, loss_bn, loss_oh, loss_adv, loss_cr], [torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)]]
        else:
            return_values = [[loss, loss_bn, loss_oh, loss_adv, loss_cr], []]

        return return_values

    def update_S(self):
        self.student.train()
        self.teacher.eval()
        
        kd_bar = tqdm(range(self.kd_steps), desc="  └── KD", leave=False)
        for i in kd_bar:     
            images = self.data_iter.next()
            images = images.cuda(self.device, non_blocking=True)

            with self.autocast():
                with torch.no_grad():
                    t_out, t_feat = self.teacher(images, return_features=True)
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
                
                loss_s = self.kl_criterion(s_out, t_out.detach())
            
            
            self.opt_S.zero_grad()
            loss_s.backward()
            self.opt_S.step()

            with torch.no_grad():
                s_outputs = s_out.max(1)[1]
                t_outputs = t_out.max(1)[1]
                correct = (s_outputs.view(-1) == t_outputs.view(-1)).sum()
                cnt = torch.numel(t_outputs)
                train_acc = (correct / cnt).detach().cpu()

            # Update history
            self.history["S_train_loss"].append(loss_s.item())
            self.history["S_train_accuracy"].append(train_acc.item())

        return loss_s, train_acc