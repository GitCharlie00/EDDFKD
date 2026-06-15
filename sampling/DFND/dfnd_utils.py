import torch
import torch.nn.functional as F

def noisy(noise_adaptation,teacher_acc,n_classes):
    noise_adaptation_softmax = F.softmax(noise_adaptation,dim=1) * (1 - teacher_acc)
    noise_adaptation_layer = torch.zeros(n_classes,n_classes)
    for i in range(n_classes):
        if i == 0:
            noise_adaptation_layer[i] = torch.cat([teacher_acc,noise_adaptation_softmax[i][i:]])
        if i == n_classes-1:
            noise_adaptation_layer[i] = torch.cat([noise_adaptation_softmax[i][:i],teacher_acc])
        else:
            noise_adaptation_layer[i] = torch.cat([noise_adaptation_softmax[i][:i],teacher_acc,noise_adaptation_softmax[i][i:]])
    return noise_adaptation_layer.cuda()

