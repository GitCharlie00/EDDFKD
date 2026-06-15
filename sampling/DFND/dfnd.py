import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from sampling.loss import kdloss
from sampling.DFND.dfnd_utils import noisy
from sampling.utils_sam import identify_outlier

class DFND():
    def __init__(self,args,ood_train_dataset,teacher,student,scheduler_S,opt_S,teacher_acc,history,method_labels):
        # From parameters
        self.args = args
        self.ood_train_dataset = ood_train_dataset
        self.teacher = teacher
        self.student = student
        self.scheduler_S = scheduler_S
        self.opt_S = opt_S
        self.teacher_acc = teacher_acc
        self.history = history
        self.method_labels = method_labels
        
        # From args
        self.lr_N = args.lr_N
        self.num_select = args.num_select
        self.num_classes = args.num_classes
        self.num_workers = args.num_workers
        self.batch_size = args.train_batch_size
        self.log_interval = args.log_interval
        self.device = args.gpu

        # Method specific
        self.data_train_loader_no_shuffle = DataLoader(self.ood_train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        value, pred = identify_outlier(self.teacher,self.data_train_loader_no_shuffle)
        positive_index = value.topk(self.num_select,largest=False)[1]
        positive_index = positive_index.tolist()

        self.outlier_criterion = nn.CrossEntropyLoss(reduction='none')
        self.nll = nn.NLLLoss().cuda()
        self.data_train_select = torch.utils.data.Subset(self.ood_train_dataset, positive_index)
        self.ood_loader_select = torch.utils.data.DataLoader(self.data_train_select, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        self.noise_adaptation = torch.nn.Parameter(torch.zeros(self.num_classes,self.num_classes-1))
        self.opt_N = optim.Adam([self.noise_adaptation], lr=self.lr_N)

    def train_loop(self):
        self.student.train()

        train_bar = tqdm(enumerate(self.ood_loader_select), total=len(self.ood_loader_select), desc="Training", leave=True)
        for step, (images, _, _) in train_bar:
            images = Variable(images).cuda(self.device)

            self.opt_S.zero_grad()
            self.opt_N.zero_grad()

            output = self.student(images)
            
            output_t = self.teacher(images).detach()
            pred = output_t.data.max(1)[1]
                
            kd_loss = kdloss(output, output_t)         
                    
            output_s = F.softmax(output, dim=1)
            output_s_adaptation = torch.matmul(output_s, noisy(self.noise_adaptation,self.teacher_acc,self.num_classes))
            ce_loss = self.nll(torch.log(output_s_adaptation), pred)
            loss = kd_loss + ce_loss

            losses = [kd_loss,ce_loss]

            self.history["N_total_loss"].append(loss.item())
            self.history["N_KDL_loss"]["values"].append(kd_loss.item())
            self.history["N_CE_loss"]["values"].append(ce_loss.item())

            # Student prediction vs pseudo-label
            student_pred = output.argmax(dim=1)
            correct = (student_pred == pred).sum().item()
            accuracy = correct / images.size(0)

            # Optionally: student CE loss (without adaptation) for monitoring
            student_ce_loss = F.cross_entropy(output, pred)

            loss.backward()
            self.opt_S.step()
            self.opt_N.step()

            self.scheduler_S.step()
            current_lr_S = self.opt_S.param_groups[0]['lr']

            if (step + 1) % self.log_interval == 0:
                postfix_dict = {}
                postfix_dict["L_tot"] = loss.item()
    
                for i, label in enumerate(self.method_labels):
                    postfix_dict[label] = losses[i].item()

                postfix_dict["L_KD"] = student_ce_loss.item()
                postfix_dict["acc_KD"] = accuracy
                postfix_dict["lr_S"] = current_lr_S

                train_bar.set_postfix(postfix_dict)