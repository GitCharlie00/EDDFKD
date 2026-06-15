import torch.nn as nn
import torchvision.models as models

from networks.resnet import ResNet34
from networks.vgg import VGG11
from networks.wrn import wrn_40_2

def get_teacher_model(name, num_classes, feature_extractor=False):
    """
    Return a PyTorch model configured for `num_classes`.
    """
    key = name.lower()
    if key == "resnet-34":
        model_name = "ResNet34"
        model = ResNet34(
            num_classes=num_classes, 
            feature_extractor=feature_extractor
        )
 
        return model, model_name

    elif key == "vgg-11":
        model_name = "VGG11"
        model = VGG11(
            num_classes=num_classes,
            batch_norm=True,
            feature_extractor=feature_extractor
        )
        
        return model, model_name
    
    elif key == "wrn-40-2":
        model_name = "WRN-40-2"
        model = wrn_40_2(
            num_classes=num_classes,
            feature_extractor=feature_extractor
        )

        return model, model_name
        
    else:
        raise ValueError(f"Unknown network '{name}'. Available: resnet-34, vgg-11, wrn-40-2")
