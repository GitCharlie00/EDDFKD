import torch.nn as nn
import torchvision.models as models

from networks.resnet import ResNet18
from networks.wrn import wrn_16_1, wrn_16_2, wrn_40_1

def get_student_model(name, num_classes):
    """
    Return a PyTorch student model configured for `num_classes`.
    Supports: resnet18, wrn-16-1, wrn-40-1, wrn-16-2
    """
    key = name.lower()

    if key == "s_resnet-18":
        model_name = "ResNet18"
        model = ResNet18(num_classes=num_classes)
        
        return model, model_name

    elif key == "s_wrn-16-1":
        model_name = "WRN-16-1"
        model = wrn_16_1(num_classes=num_classes)
        
        return model, model_name

    elif key == "s_wrn-16-2":
        model_name = "WRN-16-2"
        model = wrn_16_2(num_classes=num_classes)
        
        return model, model_name
    
    elif key == "s_wrn-40-1":
        model_name = "WRN-40-1"
        model = wrn_40_1(num_classes=num_classes)
        
        return model, model_name

    else:
        raise ValueError(f"Unknown student network '{name}'. Available: resnet, wrn-16-1, wrn-40-1, wrn-16-2")