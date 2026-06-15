import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import torch

import matplotlib.image as mpimg
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from datasets_local.dataloaders import init_train_test_loader
from datasets_local.datasets import SyntheticDataset
from networks.t_networks import get_teacher_model
from networks.s_networks import get_student_model
from utils import TextFormatter, estimate_energy_from_bn_stats, estimate_energy_theoretical, extract_features, plot_tsne, plot_umap, plot_energy, save_image_to_pdf

def plot(args, approach_suffix, user_suffix):
    ### ----------------------------------- Teacher model ------------------------------------ ###
    torch.cuda.empty_cache()
    
    # FIX: Usa sempre t_network per il teacher per coerenza
    teacher_arg = args.t_network if args.t_network else args.network
    feat_extr   = False if not args.calc_target else args.feature_extr
    
    teacher, teacher_name = get_teacher_model(teacher_arg, args.num_classes, feat_extr)
    
    # Carichiamo lo studente solo se non stiamo calcolando solo il target (opzionale, ma pulito)
    if not args.calc_target:
        _, student_name = get_student_model(args.s_network, args.num_classes)
    else:
        student_name = "None"

    formatter = TextFormatter()
    teach_str = formatter.format(
        f"Initializing teacher network {teacher_name}",
        color = "blue", 
        style = ["bold"], 
        separator = True
    )
    print(teach_str)
    
    pattern = f"{args.save_path}{args.dataset}/{teacher_name.lower()}_{args.dataset.lower()}_best_*.pth"
    matches = glob.glob(pattern)

    if len(matches) == 0:
        raise FileNotFoundError(f"No checkpoint found at: {pattern}")
                    
    checkpoint_path = matches[0]

    try:
        teacher.load_state_dict(torch.load(checkpoint_path, map_location=args.gpu)["state_dict"])
    except:
        teacher.load_state_dict(torch.load(checkpoint_path, map_location=args.gpu))
    teacher.to(args.gpu)

    print(f"[LOG] Teacher model successfully loaded from {matches[0]}")
    
    ### ---------------------------------- Dataset loading ---------------------------------- ###
    formatter = TextFormatter()
    data_str = formatter.format(
        f"Initializing dataset {args.dataset} (REAL Test Set)",
        color = "blue", 
        style = ["bold"], 
        separator = True
    )
    print(data_str)

    # Nota: real_loader è il Test Loader (dati puliti)
    _, real_loader, _, test_Dataset = init_train_test_loader(
        dataset_type=args.dataset,
        dataset_root=args.dataset_root,
        train_batch=args.train_batch_size,
        test_batch=args.test_batch_size,
        num_workers=args.num_workers
    )
    
    # Logica per i path
    if not args.calc_target:
        dataset_save_dir = f"{args.dataset_root}{teacher_name}_{student_name}_{args.method.lower()}{approach_suffix}_{args.dataset}{user_suffix}"
        filename_out = f"work/project/plots/{args.method.lower()}_{args.dataset}_{args.t_network.lower()}_{args.s_network.lower()}{approach_suffix}{user_suffix}.png"

        synth_dataset = SyntheticDataset(root=dataset_save_dir, dataset=args.dataset)
        synth_loader = DataLoader(synth_dataset, batch_size=512, shuffle=False, num_workers=4)
        
        real_images_path = f"{args.dataset_root}temp_{args.method.lower()}{approach_suffix}_{args.dataset}/"
        os.makedirs(real_images_path, exist_ok=True)

        print(f"[LOG] Exporting {args.dataset} as PNGs for metrics computation")
        counter = 0
        total = len(test_Dataset) # O limitati a 100 per velocità
        for batch in tqdm(real_loader, desc="Saving Real images"):
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            for img in images:
                save_image(img, os.path.join(real_images_path, f'{counter:06d}.png'))
                counter += 1
                if counter >= total: break
            if counter >= total: break
    else:
        # Se calcoliamo solo il target, salviamo un plot semplice
        filename_out = f"work/project/plots/target_energy_{teacher_name.lower()}_{args.dataset}.png"

    ### -------------------------------- Metrics computation --------------------------------- ###
    formatter = TextFormatter()
    comp_str = formatter.format(
        f"Starting metrics computation",
        color = "blue", 
        style = ["bold"], 
        separator = True
    )
    print(comp_str)

    # --- RAMO 1: CALCOLO SOLO TARGET ---
    if args.calc_target:
        print(f"[LOG] Computing Energy statistics on REAL data ({args.dataset})...")
        X_real, energy_real = extract_features(real_loader, args.gpu, teacher, "REAL",feat_extr)

        # Calcolo statistiche
        real_energy_mean = energy_real.mean().item()
        real_energy_std = energy_real.std().item()
        
        target_info = (
            f"\n{'='*40}\n"
            f"   ENERGY STATISTICS (Real Data)\n"
            f"{'='*40}\n"
            f" Model:   {teacher_name}\n"
            f" MEAN (Target): {real_energy_mean:.4f}\n"
            f" STD:           {real_energy_std:.4f}\n"
            f"{'='*40}\n"
        )
        print(target_info)

        print("[LOG] Plot Energy (Single Distribution)")
        estimated_energy_means = {}
        
        estimated_energy_means["BN"] = estimate_energy_from_bn_stats(teacher)
        estimated_energy_means["Theoretical"] = estimate_energy_theoretical(teacher,args.num_classes)

        plot_energy(energy_real, [], filename_out, args.dataset.upper(), "Real samples Only",args.calc_target, real_energy_mean, estimated_energy_means)

        print(f"[SAVED] Target Energy plot saved to {filename_out}")

    # --- RAMO 2: COMPARAZIONE COMPLETA ---
    else:
        print(f"[LOG] Proceeding with Synthetic Data comparison...")
        
        # 1. Estrai Features Reali
        X_real, energy_real = extract_features(real_loader, args.gpu, teacher, "REAL",feat_extr)
        # FIX: Calcola la media qui per usarla nel gap
        real_energy_mean = energy_real.mean().item()

        # 2. Estrai Features Sintetiche
        X_synth, energy_synth = extract_features(synth_loader, args.gpu, teacher, "SYNTHETIC",feat_extr)
        synth_energy_mean = energy_synth.mean().item()

        # 3. Calcola Gap
        energy_gap = abs(real_energy_mean - synth_energy_mean)
        print(f"[LOG] Real Mean: {real_energy_mean:.4f} | Synthetic Mean: {synth_energy_mean:.4f}")
        print(f"[LOG] Energy Gap: {energy_gap:.4f}")

        # 4. Plots
        print("[LOG] Plot t-SNE")
        plot_tsne(X_real, X_synth, "tsne.png", args.dataset.upper(), args.method.upper() + " Synthetic samples")

        print("[LOG] Plot UMAP")
        plot_umap(X_real, X_synth, "umap.png", args.dataset.upper(), args.method.upper() + " Synthetic samples")

        print("[LOG] Plot Energy")
        plot_energy(energy_real, energy_synth, "energy.png", args.dataset.upper(), args.method.upper() + " Synthetic samples")

        # 5. Combina le immagini
        fig, axes = plt.subplots(3, 1, figsize=(8, 14)) 

        title_text = (
            f"REAL Energy Mean: {real_energy_mean:.2f} | SYNTH Energy Mean: {synth_energy_mean:.2f}\n"
            f"GAP (Lower is better): {energy_gap:.4f}"
        )
        
        fig.suptitle(title_text, fontsize=14, weight='bold')

        for ax, path in zip(
            axes,
            ["tsne.png", "umap.png", "energy.png"]    
        ):
            if os.path.exists(path):
                img = mpimg.imread(path)
                ax.imshow(img)
                ax.axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(filename_out)
        plt.close()

        # Rimuove i PNG temporanei
        for f in ["tsne.png", "umap.png", "energy.png"]:
            if os.path.exists(f):
                os.remove(f)
        
        if os.path.exists(real_images_path):
            shutil.rmtree(real_images_path)
            
        print(f"[SAVED] Comparison plot saved to {filename_out}")