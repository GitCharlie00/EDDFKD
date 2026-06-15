import os
import random
import torch

from collections import defaultdict
from PIL import Image
from torchvision import transforms

class DataIter(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iter = iter(self.dataloader)
    
    def next(self):
        try:
            data = next( self._iter )
        except StopIteration:
            self._iter = iter(self.dataloader)
            data = next( self._iter )
        return data

def collect_sample_images(dataset_save_dir, num_samples=10):
    """
    Raccoglie un campione casuale di immagini generate per il report.
    
    Args:
        dataset_save_dir (str): Directory dove sono salvate le immagini generate
        num_samples (int): Numero di immagini da campionare
    
    Returns:
        list: Lista di tuple (image_path, pseudo_label)
    """
    
    sample_images = []
    
    try:
        # Ottieni tutti i file PNG nella directory
        all_files = [f for f in os.listdir(dataset_save_dir) if f.endswith('.png')]
        
        if len(all_files) == 0:
            print("[WARNING] No PNG files found in dataset directory")
            return sample_images
        
        # Campiona casualmente
        sampled_files = random.sample(all_files, min(num_samples, len(all_files)))
        
        for filename in sampled_files:
            # Estrai la pseudo-label dal nome del file (formato: XXXXXX_Y.png)
            try:
                parts = filename.split('_')
                if len(parts) >= 2:
                    pseudo_label = int(parts[1].split('.')[0])  # Rimuovi l'estensione .png
                    image_path = os.path.join(dataset_save_dir, filename)
                    sample_images.append((image_path, pseudo_label))
            except (ValueError, IndexError):
                print(f"[WARNING] Could not parse label from filename: {filename}")
                continue
        
        print(f"[LOG] Collected {len(sample_images)} sample images for report")
        
    except Exception as e:
        print(f"[ERROR] Error collecting sample images: {e}")
    
    return sample_images

def identify_outlier(teacher,data_train_loader_noshuffle):
    value = []
    pred_list = []
    index = 0
    celoss = torch.nn.CrossEntropyLoss(reduction = 'none')
    
    teacher.eval()
    for i,(inputs, labels,_) in enumerate(data_train_loader_noshuffle):
        inputs = inputs.cuda()
        outputs = teacher(inputs)
        pred = outputs.data.max(1)[1]
        loss = celoss(outputs, pred)
        value.append(loss.detach().clone())
        index += inputs.shape[0]
        pred_list.append(pred)
    return torch.cat(value,dim=0), torch.cat(pred_list,dim=0) 

def collect_sample_images_from_test(test_loader, teacher, device, num_samples=10, save_dir="sample_test_images", dataset="cifar10"):
    """
    Estrae le prime `num_samples` immagini dal test_loader, ottiene le pseudo-label dal teacher
    e le salva in `save_dir`. Ritorna una lista (image_path, pseudo_label) compatibile con il report.

    Args:
        test_loader: DataLoader test set
        teacher: modello teacher (valutazione)
        device: CUDA o CPU
        num_samples: quante immagini salvare
        save_dir: cartella dove salvare le immagini
        dataset: 'cifar10' o 'cifar100' (per denormalizzazione)

    Returns:
        sample_images: list di (image_path, pseudo_label)
        label_count: dict con count delle pseudo-label
    """
    os.makedirs(save_dir, exist_ok=True)
    sample_images = []
    label_count = defaultdict(int)

    # Valori di denormalizzazione
    if dataset.lower() == "cifar10":
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    elif dataset.lower() == "cifar100":
        mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1)
        std = torch.tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1)
    else:
        raise ValueError(f"Dataset '{dataset}' non supportato.")

    to_pil = transforms.ToPILImage()

    teacher.eval()
    with torch.no_grad():
        # Prendi il primo batch
        images, _, _ = next(iter(test_loader))
        images = images[:num_samples].to(device)

        # Ottieni pseudo-label dal teacher
        outputs = teacher(images)
        pseudo_labels = torch.argmax(outputs, dim=1)

        for i in range(images.size(0)):
            img = images[i].cpu()
            label = pseudo_labels[i].item()
            label_count[label] += 1

            # Denormalizza immagine
            img_denorm = img * std + mean
            img_denorm = torch.clamp(img_denorm, 0, 1)

            # Salva immagine come PNG
            img_pil = to_pil(img_denorm)
            filename = f"{i:02d}_{label}.png"
            path = os.path.join(save_dir, filename)
            img_pil.save(path)

            sample_images.append((path, label))

    return sample_images, label_count