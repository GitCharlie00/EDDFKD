import torch
from torch.utils.data import DataLoader

from datasets_local.datasets import *

def init_train_test_loader(dataset_type, dataset_root, train_batch, test_batch, num_workers, mosaick_ood=False, dfnd_acc=False):
    # --- CIFAR-10 ---
    if dataset_type == "cifar10":
        Dataset_class = CIFAR10Dataset
        # dts_root_path non viene usato nel tuo codice originale, ma lo lascio per coerenza
        dts_root_path = dataset_root + 'CIFAR10/' 
        teacher_acc = torch.tensor([0.9523])

    # --- CIFAR-100 ---
    elif dataset_type == "cifar100":
        Dataset_class = CIFAR100Dataset
        dts_root_path = dataset_root + 'CIFAR100/'
        teacher_acc = torch.tensor([0.7774])

    # --- SVHN ---
    elif dataset_type == "svhn":
        Dataset_class = SVHNDataset
        dts_root_path = dataset_root + 'SVHN/'
        teacher_acc = torch.tensor([0.9600]) # Placeholder accuracy

    # --- Tiny-ImageNet ---
    elif dataset_type == "tiny_imagenet":
        Dataset_class = TinyImageNetDataset
        dts_root_path = dataset_root + 'tiny-imagenet-200/'
        teacher_acc = torch.tensor([0.6500]) # Placeholder accuracy

    # --- SUN397 ---
    elif dataset_type == "sun":
        Dataset_class = SUNDataset
        dts_root_path = dataset_root + 'SUN397/'
        teacher_acc = torch.tensor([0.6000]) # Placeholder accuracy

    # --- Places365 ---
    elif dataset_type == "places365":
        Dataset_class = Places365Dataset
        dts_root_path = dataset_root + 'Places365/'
        teacher_acc = torch.tensor([0.5500]) # Placeholder accuracy

    elif dataset_type == "imagenet":
        Dataset_class = ImageNetDataset
        dts_root_path = dataset_root + 'imagenet/' # PyTorch si aspetta spesso una sottocartella
        teacher_acc = torch.tensor([0.7615]) # Accuracy tipica ResNet50
    else:
        raise ValueError(f"[!!!ERROR!!!] Unsupported dataset type: {dataset_type}")

    # Load Datasets
    training_Dataset = Dataset_class(
        dataset_root=dataset_root, 
        train=True,
        mosaick_ood=mosaick_ood
    )
    test_Dataset = Dataset_class(
        dataset_root=dataset_root, 
        train=False
    )
    
    # Create Dataloaders
    training_DataLoader = DataLoader(
        training_Dataset, batch_size=train_batch, shuffle=True, pin_memory=True, num_workers=num_workers
    )
    test_DataLoader = DataLoader(
        test_Dataset, batch_size=test_batch, shuffle=False, pin_memory=True, num_workers=num_workers
    )

    if not dfnd_acc:
        return training_DataLoader, test_DataLoader, training_Dataset, test_Dataset
    else:
        return training_DataLoader, test_DataLoader, training_Dataset, test_Dataset, teacher_acc

