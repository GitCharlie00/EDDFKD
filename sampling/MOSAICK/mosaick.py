import torch
import torch.nn.functional as F

from tqdm import tqdm

from sampling.MOSAICK.mosaick_utils import Normalizer, dummy_ctx
from sampling.loss import kldiv

class MOSAICK():
    def __init__(self,args,models,optims,schedulers,loaders,history,method_labels):
        # From parameters
        self.args = args
        self.teacher = models[0]
        self.student = models[1]
        self.G = models[2]
        self.D = models[3]

        self.opt_S = optims[0]
        self.opt_G = optims[1]
        self.opt_D = optims[2]

        self.sched_S = schedulers[0]
        self.sched_G = schedulers[1]
        self.sched_D = schedulers[2]

        self.train_loader = loaders[0]
        self.ood_iter = loaders[1]

        self.history = history
        self.method_labels = method_labels

        # From args
        self.batch_size = args.train_batch_size
        self.z_dim = args.z_dim
        self.local = args.local
        self.adv = args.adv
        self.align = args.align
        self.log_interval = args.log_interval
        self.kd_steps = args.kd_steps
        self.T = args.temperature
        self.device = args.gpu

        # Method specific
        self.autocast = dummy_ctx
        self.normalizer = Normalizer(args.dataset)

    def train_loop(self):
        self.student.train()
        self.teacher.eval()
        self.D.train()
        self.G.train()

        train_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Generating", leave=True)
        for step, (real, _, _) in train_bar:
            loss_d, images = self.update_PD(real)
            loss_g, loss_adv, loss_align, loss_local = self.update_G(images)
            loss_s, train_acc = self.update_S()

            losses = [loss_g,loss_adv,loss_align,loss_local,loss_d,loss_s]

            self.sched_S.step()
            current_lr_S = self.opt_S.param_groups[0]['lr']
            self.sched_G.step()
            current_lr_G = self.opt_G.param_groups[0]['lr']
            self.sched_D.step()
            current_lr_D = self.opt_D.param_groups[0]['lr']

            if (step + 1) % self.log_interval == 0:
                postfix_dict = {}
                postfix_dict["L_G"] = losses[0].item()
    
                for i, label in enumerate(self.method_labels):
                    postfix_dict[label] = losses[i+1].item()

                postfix_dict["L_D"] = losses[-2].item()
                postfix_dict["L_KD"] = losses[-1].item()
                postfix_dict["acc_KD"] = train_acc.item()
                postfix_dict["lr_S"] = current_lr_S
                postfix_dict["lr_G"] = current_lr_G
                postfix_dict["lr_D"] = current_lr_D

                train_bar.set_postfix(postfix_dict)

    def update_PD(self,real):
        real = real.cuda(self.device, non_blocking=True)
        
        with self.autocast():
            z = torch.randn(size=(self.batch_size, self.z_dim), device=self.device)
            images = self.G(z)
            images = self.normalizer(images)
            d_out_fake = self.D(images.detach())
            d_out_real = self.D(real.detach())

            loss_d_1 = F.binary_cross_entropy_with_logits(
                d_out_fake, 
                torch.zeros_like(d_out_fake), 
                reduction='sum'
            )  
            loss_d_2 = F.binary_cross_entropy_with_logits(
                d_out_real, 
                torch.ones_like(d_out_real), 
                reduction='sum'
            )

            loss_d = ((loss_d_1 + loss_d_2) / (2*len(d_out_fake))) * self.local
                    
        self.opt_D.zero_grad()
        loss_d.backward()
        self.opt_D.step()
        self.history["D_train_loss"].append(loss_d.item())

        return loss_d, images
    
    def update_G(self,images):
        with self.autocast():
            t_out = self.teacher(images)
            s_out = self.student(images)

            pyx = F.softmax(t_out, dim = 1)
            log_softmax_pyx = F.log_softmax(t_out, dim=1)
            py = pyx.mean(0)

            d_out_fake = self.D(images)
            loss_local = F.binary_cross_entropy_with_logits(d_out_fake, torch.ones_like(d_out_fake), reduction='sum') / len(d_out_fake)
            loss_align = -(pyx * log_softmax_pyx).sum(1).mean()
            loss_adv = -kldiv(s_out, t_out)
            #### se serve qui stava balance

            loss_g = loss_adv * self.adv + loss_align * self.align + self.local * loss_local

        self.opt_G.zero_grad()        
        loss_g.backward()
        self.opt_G.step()

        self.history["G_train_loss"].append(loss_g.item())
        self.history["G_ADV_train_loss"]["values"].append(loss_adv.item())
        self.history["G_ALG_train_loss"]["values"].append(loss_align.item())
        self.history["G_LOC_train_loss"]["values"].append(loss_local.item())
        
        return loss_g, loss_adv, loss_align, loss_local

    def update_S(self):
        kd_bar = tqdm(range(self.kd_steps), desc="  └── KD", leave=False)
        for i in kd_bar:
            with self.autocast():
                with torch.no_grad():
                    z = torch.randn(size=(self.batch_size, self.z_dim), device=self.device)
                    vis_images = images = self.G(z)
                    images = self.normalizer(images)
                    ood_images = self.ood_iter.next()[0].to(self.device)
                    images = torch.cat([images, ood_images])
                    t_out = self.teacher(images)
                
                s_out = self.student(images.detach())
                loss_s = kldiv(s_out, t_out.detach(), T=self.T)
                self.history["S_train_loss"].append(loss_s.item())

                with torch.no_grad():
                    s_outputs = s_out.max(1)[1]
                    t_outputs = t_out.max(1)[1]
                    correct = (s_outputs.view(-1) == t_outputs.view(-1)).sum()
                    cnt = torch.numel(t_outputs)
                    train_acc = (correct / cnt).detach().cpu()
                    self.history["S_train_accuracy"].append(train_acc)
            
            self.opt_S.zero_grad()
            loss_s.backward()
            self.opt_S.step()

        return loss_s, train_acc

