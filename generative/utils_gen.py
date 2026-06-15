import numpy as np
import os
import random
import torch
import torch.nn.functional as F

from contextlib import contextmanager
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.utils.data import Subset, DataLoader

def normalize(tensor, mean, std, reverse=False):
    if reverse:
        _mean = [ -m / s for m, s in zip(mean, std) ]
        _std = [ 1/s for s in std ]
    else:
        _mean = mean
        _std = std
    
    _mean = torch.as_tensor(_mean, dtype=tensor.dtype, device=tensor.device)
    _std = torch.as_tensor(_std, dtype=tensor.dtype, device=tensor.device)
    tensor = (tensor - _mean[None, :, None, None]) / (_std[None, :, None, None])
    return tensor

class Normalizer(object):
    def __init__(self, dataset):
        if dataset == "cifar10":
            self.mean = (0.4914, 0.4822, 0.4465)
            self.std = (0.2023, 0.1994, 0.2010)
        else:
            self.mean = (0.5071, 0.4867, 0.4408)
            self.std = (0.2675, 0.2565, 0.2761)

    def __call__(self, x, reverse=False):
        return normalize(x, self.mean, self.std, reverse=reverse)

@contextmanager
def dummy_ctx(*args, **kwds):
    try:
        yield None
    finally:
        pass

def _get_label_distribution(sample_images):
    label_count = {}
    for img_path, pseudo_label in sample_images:
        if pseudo_label not in label_count:
            label_count[pseudo_label] = 0
        label_count[pseudo_label] += 1
    
    return label_count

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

# def get_confounder_dict(M,confounder_size,pca=False):
#     # Spostiamo M su CPU e in numpy per KMeans
#     M_np = M.detach().cpu().numpy()

#     # (Opzionale) PCA, come nel paper
#     if pca:
#         pca_model = PCA(n_components=min(M_np.shape[1], 50))  # es: riduci a 50 dims
#         M_np = pca_model.fit_transform(M_np)

#     # Applica KMeans++
#     kmeans = KMeans(n_clusters=confounder_size, init='k-means++', n_init=10, random_state=42)
#     cluster_labels = kmeans.fit_predict(M_np)

#     # Calcola centroide z_i in spazio originale (non ridotto da PCA)
#     # (Se hai fatto PCA, puoi usare i centroidi nello spazio PCA o proiettarli indietro)
#     centroids = []
#     Ps = []
#     for i in range(confounder_size):
#         mask = (cluster_labels == i)
#         count = mask.sum()
#         if count == 0:
#             # nessun punto in questo cluster -> usa il centroide di kmeans stesso
#             centroid = torch.tensor(kmeans.cluster_centers_[i], dtype=M.dtype)
#             ps_val = 0.0
#         else:
#             points = M[mask]
#             centroid = points.mean(dim=0)
#             ps_val = count / len(M_np)
#         centroids.append(centroid)
#         Ps.append(ps_val)

#     Z = torch.stack(centroids, dim=0).to(M.device)  # shape [confounder_size, d]
#     Ps = torch.tensor(Ps, dtype=M.dtype, device=M.device)  # shape [confounder_size]

#     return Z, Ps

def get_confounder_dict(M, confounder_size, pca=False):
    """
    Versione ottimizzata PyTorch-nativa di KMeans per evitare il collo di bottiglia CPU.
    M: Tensor [batch_size, feature_dim] già su GPU
    """
    # Nota: PCA è lento e spesso non necessario se le dim non sono enormi.
    # Se vuoi mantenerlo, dovresti usare torch.pca_lowrank, ma rallenta.
    # Nel paper usano PCA opzionale, qui lo omettiamo per velocità pura se non strettamente richiesto.
    
    # M è già su GPU, non spostarlo su CPU!
    x = M 
    batch_size, dims = x.shape
    
    # 1. Inizializzazione casuale dei centroidi (Forgy method)
    # Scegliamo indici casuali dal batch attuale
    if batch_size > confounder_size:
        indices = torch.randperm(batch_size, device=x.device)[:confounder_size]
        centroids = x[indices]
    else:
        # Fallback se il batch è minuscolo (raro)
        centroids = x[:confounder_size]

    # 2. Iterazioni KMeans (bastano poche iterazioni per stabilizzare i cluster su feature maps)
    n_iter = 10 
    
    for _ in range(n_iter):
        # Calcola distanze (broadcasting ottimizzato)
        # x: [B, D], centroids: [K, D] -> dist: [B, K]
        # ||a-b||^2 = a^2 + b^2 - 2ab
        
        # Modo efficiente con torch.cdist
        dists = torch.cdist(x, centroids)
        
        # Assegna cluster
        labels = torch.argmin(dists, dim=1)
        
        # Aggiorna centroidi
        new_centroids = []
        counts = []
        
        for i in range(confounder_size):
            mask = (labels == i)
            if mask.any():
                cluster_points = x[mask]
                new_centroids.append(cluster_points.mean(dim=0))
                counts.append(mask.sum().float())
            else:
                # Gestione cluster vuoti: mantieni il vecchio o ri-inizializza
                new_centroids.append(centroids[i])
                counts.append(torch.tensor(0., device=x.device))
        
        centroids = torch.stack(new_centroids)
        
    # Calcolo proporzioni finali
    counts = torch.stack(counts)
    total = counts.sum()
    Ps = counts / (total + 1e-6) # Evita div by zero
    
    # Ritorna Z e Ps direttamente su GPU
    return centroids, Ps

def apply_datafree(args,epoch,dfkd_method,warmup_printed):
    if args.method == "dafl":
        G_losses, S_losses = dfkd_method.dafl_loop(epoch)
    else:
        # Update G
        G_losses = dfkd_method.update_G(epoch)

        # Update student via KD
        if args.method == "fast":
            if epoch >= args.warmup:
                if not warmup_printed:
                    print(f"[LOG] Warmup ended - Starting to KD")
                    warmup_printed = True
                S_losses = dfkd_method.update_S(epoch)
            else:
                S_losses = [torch.tensor(0.0),torch.tensor(0.0)]
        else:
            S_losses = dfkd_method.update_S(epoch)
    
    return G_losses, S_losses, warmup_printed