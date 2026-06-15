import io
import os
import torch
import torchvision
import torchvision.transforms as transforms
import urllib.request
import zipfile

from contextlib import redirect_stdout
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from datasets_local.data_augmentation import Cutout

class CIFAR10Dataset(Dataset):
    def __init__(self, dataset_root, train=True, download=True, transform=None, mosaick_ood=False):
        self.dataset_root = dataset_root
        self.train = train
        self.download = download

        if train and not mosaick_ood:
            self.transform = transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ])
        else:
            self.transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ])

        buf = io.StringIO()
        with redirect_stdout(buf):
            self.dataset = torchvision.datasets.CIFAR10(
                root=self.dataset_root,
                train=self.train,
                download=self.download,
                transform=self.transform
            )
        for line in buf.getvalue().splitlines():
            if "Files already downloaded and verified" in line:
                print(f"[LOG] {line}")
            else:
                print(line)

        self.classes = self.dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_tensor, numeric_label = self.dataset[idx]
        text_label = self.classes[numeric_label]

        return img_tensor, numeric_label, text_label

class CIFAR100Dataset(Dataset):
    def __init__(self, dataset_root, train=True, download=True, transform=None, mosaick_ood=False):
        self.dataset_root = dataset_root
        self.train = train
        self.download = download

        if train and not mosaick_ood:
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
      
        # Suppress and log any output from torchvision
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.dataset = torchvision.datasets.CIFAR100(
                root=self.dataset_root,
                train=self.train,
                download=self.download,
                transform=self.transform
            )
        for line in buf.getvalue().splitlines():
            if "Files already downloaded and verified" in line:
                print(f"[LOG] {line}")
            else:
                print(line)

        self.classes = self.dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_tensor, numeric_label = self.dataset[idx]
        text_label = self.classes[numeric_label]

        return img_tensor, numeric_label, text_label

class SVHNDataset(Dataset):
    def __init__(self, dataset_root, train=True, download=True, transform=None, mosaick_ood=False):
        self.dataset_root = dataset_root
        self.train = train
        self.download = download

        # Mapping del booleano 'train' allo string 'split' di SVHN
        self.split = 'train' if train else 'test'

        # Statistiche specifiche per SVHN
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)

        if train and not mosaick_ood:
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                # NOTA: RandomHorizontalFlip rimosso perché specchiare i numeri (es. 3, 7) è errato
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        # Gestione output silenzioso come nella tua classe CIFAR
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.dataset = torchvision.datasets.SVHN(
                root=self.dataset_root,
                split=self.split,
                download=self.download,
                transform=self.transform
            )
        
        # Filtra e stampa solo i messaggi rilevanti
        for line in buf.getvalue().splitlines():
            if "Using downloaded and verified file" in line or "Downloading" in line:
                print(f"[LOG] {line}")
            else:
                # SVHN è spesso molto verboso nel download, stampiamo se non è rumore
                if line.strip(): 
                    print(line)

        # SVHN non ha .classes nativo, lo creiamo noi (le cifre 0-9)
        self.classes = [str(i) for i in range(10)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # SVHN restituisce (image, label) dove label è int
        img_tensor, numeric_label = self.dataset[idx]
        
        # Gestione label: a volte SVHN restituisce tensori o int, forziamo int
        if isinstance(numeric_label, torch.Tensor):
            numeric_label = numeric_label.item()
            
        text_label = self.classes[numeric_label]

        return img_tensor, numeric_label, text_label

class TinyImageNetDataset(Dataset):
    def __init__(self, dataset_root, train=True, download=True, transform=None, mosaick_ood=False):
        self.dataset_root = dataset_root
        self.base_path = os.path.join(dataset_root, 'tiny-imagenet-200')
        self.train = train
        self.url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        
        # Statistiche standard calcolate su TinyImageNet
        # Mean: [0.4802, 0.4481, 0.3975], Std: [0.2302, 0.2265, 0.2262]
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2302, 0.2265, 0.2262)

        # Download e Estrazione
        if download:
            self._download_and_extract()

        if not os.path.exists(self.base_path):
            raise RuntimeError("Dataset not found. Please set download=True")

        # Setup Transforms
        if train and not mosaick_ood:
            self.transform = transforms.Compose([
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        # Caricamento Mappa Nomi (WordNet ID -> Human String)
        self.id_to_human = self._load_human_labels()

        # Caricamento Dataset
        if self.train:
            print(f"[LOG] Loading Training Set from {self.base_path}/train")
            train_dir = os.path.join(self.base_path, 'train')
            self.data = ImageFolder(root=train_dir, transform=self.transform)
            # ImageFolder usa le cartelle come classi (n01443537, ...)
            self.classes = self.data.classes 
            self.class_to_idx = self.data.class_to_idx
        else:
            print(f"[LOG] Loading Validation Set from {self.base_path}/val")
            self.images, self.targets, self.classes, self.class_to_idx = self._load_validation_set()

    def _download_and_extract(self):
        zip_path = os.path.join(self.dataset_root, 'tiny-imagenet-200.zip')
        
        if os.path.exists(self.base_path):
            print("[LOG] Tiny-ImageNet already exists.")
            return

        if not os.path.exists(self.dataset_root):
            os.makedirs(self.dataset_root)

        if not os.path.exists(zip_path):
            print(f"[LOG] Downloading Tiny-ImageNet from {self.url}...")
            urllib.request.urlretrieve(self.url, zip_path)
        
        print("[LOG] Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.dataset_root)
        print("[LOG] Extraction complete.")

    def _load_human_labels(self):
        # Legge words.txt per mappare n01443537 -> "goldfish"
        path = os.path.join(self.base_path, 'words.txt')
        mapping = {}
        if os.path.exists(path):
            with open(path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        mapping[parts[0]] = parts[1]
        return mapping

    def _load_validation_set(self):
        # La cartella val è piatta, bisogna leggere val_annotations.txt
        val_dir = os.path.join(self.base_path, 'val')
        img_dir = os.path.join(val_dir, 'images')
        annot_file = os.path.join(val_dir, 'val_annotations.txt')

        # Dobbiamo assicurare che l'ordine delle classi sia IDENTICO a quello del Training
        # Leggiamo wnids.txt che definisce l'ordine ufficiale
        wnids_path = os.path.join(self.base_path, 'wnids.txt')
        with open(wnids_path, 'r') as f:
            classes = [x.strip() for x in f.readlines()]
        
        classes.sort()

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        images = []
        targets = []

        with open(annot_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                img_name = parts[0]
                wnid = parts[1]
                
                img_path = os.path.join(img_dir, img_name)
                images.append(img_path)
                targets.append(class_to_idx[wnid])

        return images, targets, classes, class_to_idx

    def __len__(self):
        if self.train:
            return len(self.data)
        else:
            return len(self.images)

    def __getitem__(self, idx):
        if self.train:
            # ImageFolder ritorna (img, label_idx)
            img_tensor, numeric_label = self.data[idx]
            wnid = self.classes[numeric_label]
        else:
            # Validation custom loader
            img_path = self.images[idx]
            numeric_label = self.targets[idx]
            wnid = self.classes[numeric_label]
            
            # Caricamento manuale immagine
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img_tensor = self.transform(img)

        # Recupera il nome umano (es. "goldfish") o fallback su ID (es. "n01443537")
        text_label = self.id_to_human.get(wnid, wnid)

        return img_tensor, numeric_label, text_label

class SUNDataset(Dataset):
    def __init__(self, dataset_root, train=True, download=True, transform=None, mosaick_ood=False):
        self.dataset_root = dataset_root
        self.train = train
        
        # Statistiche standard ImageNet
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        # Configurazione Trasformazioni
        if train and not mosaick_ood:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        print(f"[LOG] Loading SUN397 from Hugging Face (Split: {'Train' if train else 'Test'})...")
        
        # Caricamento da Hugging Face
        # cache_dir=self.dataset_root assicura che i dati vengano salvati sul tuo SSD
        try:
            # Scarichiamo tutto il dataset (solitamente è un unico blocco 'train')
            full_dataset = load_dataset("1aurent/SUN397", split="train", cache_dir=self.dataset_root)
            
            # Poiché questo dataset HF non ha split nativi train/test, ne creiamo uno noi.
            # Usiamo seed=42 per garantire che lo split sia sempre identico ad ogni avvio.
            # (80% Train, 20% Test)
            split_dataset = full_dataset.train_test_split(test_size=0.2, seed=42)
            
            if self.train:
                self.data = split_dataset['train']
            else:
                self.data = split_dataset['test']
                
            # Recuperiamo i nomi delle classi dai metadata di HuggingFace
            self.classes = full_dataset.features['label'].names
            
        except Exception as e:
            print(f"[ERROR] Errore nel caricamento da Hugging Face: {e}")
            raise e

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Hugging Face restituisce un dizionario {'image': PIL, 'label': int}
        item = self.data[idx]
        
        image = item['image']
        numeric_label = item['label']
        
        # IMPORTANTE: Convertiamo sempre in RGB. 
        # Alcune immagini di SUN397 sono in scala di grigi e farebbero crashare la Conv2d.
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Applica le trasformazioni (ToTensor, Normalize, ecc.)
        if self.transform:
            img_tensor = self.transform(image)
        else:
            img_tensor = transforms.ToTensor()(image)

        text_label = self.classes[numeric_label]

        return img_tensor, numeric_label, text_label

class Places365Dataset(Dataset):
    def __init__(self, dataset_root, train=True, download=True, transform=None, mosaick_ood=False):
        self.dataset_root = dataset_root
        self.train = train
        
        # Statistiche standard ImageNet (usate anche per Places)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        if train and not mosaick_ood:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        split_name = 'train' if train else 'val'
        print(f"[LOG] Loading Places365 from Hugging Face (Split: {split_name})...")

        try:
            # Usiamo il dataset 'timm/places365' che è ben strutturato.
            # cache_dir=self.dataset_root salva i dati nel tuo percorso SSD/custom
            self.data = load_dataset(
                "torch-uncertainty/Places365", 
                split=split_name, 
                cache_dir=self.dataset_root
            )
            
            # Recuperiamo i nomi delle classi dai metadata
            self.classes = self.data.features['label'].names
            
        except Exception as e:
            print(f"[ERROR] Errore nel caricamento Places365 da Hugging Face: {e}")
            raise e

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Hugging Face ritorna un dict {'image': PIL, 'label': int}
        item = self.data[idx]
        
        image = item['image']
        numeric_label = item['label']
        
        # Conversione di sicurezza in RGB (come per SUN397)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        if self.transform:
            img_tensor = self.transform(image)
        else:
            img_tensor = transforms.ToTensor()(image)
            
        text_label = self.classes[numeric_label]

        return img_tensor, numeric_label, text_label

# --- ImageNet Dataset ---
class ImageNetDataset(Dataset):
    def __init__(self, dataset_root, train=True, download=False, transform=None, mosaick_ood=False):
        self.dataset_root = dataset_root
        self.train = train
        self.split = 'train' if train else 'val'
        
        # Statistiche ufficiali ImageNet
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        # Trasformazioni standard (ResNet richiede 224x224)
        if train and not mosaick_ood:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        print(f"[LOG] Loading ImageNet (Split: {self.split})...")
        
        # Tenta di caricare usando la classe ufficiale
        # Nota: Se il dataset non è scaricato, questo potrebbe fallire se download=False
        try:
            self.dataset = torchvision.datasets.ImageNet(
                root=self.dataset_root,
                split=self.split,
                download=download,
                transform=self.transform
            )
        except RuntimeError as e:
            print(f"[ERROR] ImageNet not found in {dataset_root}. Auto-download might not work for ImageNet full.")
            print("Try using ImageFolder if you have the data extracted in 'train' and 'val' folders.")
            # Fallback a ImageFolder se la struttura è manuale (root/train/class/img.jpg)
            split_dir = os.path.join(dataset_root, 'imagenet', self.split)
            if os.path.exists(split_dir):
                print(f"[LOG] Fallback to ImageFolder on {split_dir}")
                self.dataset = torchvision.datasets.ImageFolder(root=split_dir, transform=self.transform)
            else:
                raise e

        self.classes = self.dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_tensor, numeric_label = self.dataset[idx]
        
        # Gestione label stringa
        # ImageNet classes sono spesso tuple o stringhe, semplifichiamo
        if hasattr(self.dataset, 'classes'):
             text_label = str(self.dataset.classes[numeric_label])
        else:
             text_label = str(numeric_label)

        return img_tensor, numeric_label, text_label

class SyntheticDataset(Dataset):
    def __init__(self, root, dataset):
        self.image_paths = sorted([
            os.path.join(root, fname)
            for fname in os.listdir(root)
            if fname.endswith('.png')
        ])
        if dataset == "cifar10":
            self.transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2023, 0.1994, 0.2010)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408),
                                    (0.2675, 0.2565, 0.2761)),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img)