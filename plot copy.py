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

from datasets.dataloaders import init_train_test_loader
from datasets.datasets import SyntheticDataset
from networks.t_networks import get_teacher_model
from networks.s_networks import get_student_model
from utils import TextFormatter, extract_features, plot_tsne, plot_umap, plot_energy, save_image_to_pdf

def plot(args,approach_suffix):
    ### ----------------------------------- Teacher model ------------------------------------ ###
    torch.cuda.empty_cache()
    
    teacher, teacher_name = get_teacher_model(args.t_network, args.num_classes, args.feature_extr)
    _, student_name = get_student_model(args.s_network, args.num_classes)

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
        f"Initializing dataset {args.dataset} & sythetic samples",
        color = "blue", 
        style = ["bold"], 
        separator = True
    )
    print(data_str)

    _, real_loader, _, test_Dataset = init_train_test_loader(
        dataset_type=args.dataset,
        dataset_root=args.dataset_root,
        train_batch=args.train_batch_size,
        test_batch=args.test_batch_size,
        num_workers=args.num_workers
    )
    
    dataset_save_dir = f"{args.dataset_root}{teacher_name}_{student_name}_{args.method.lower()}{approach_suffix}_{args.dataset}"
    
    filename_out = f"work/project/plots/{args.method.lower()}_{args.dataset}_{args.t_network.lower()}_{args.s_network.lower()}{approach_suffix}.png"

    synth_dataset = SyntheticDataset(root=dataset_save_dir, dataset=args.dataset)
    synth_loader = DataLoader(synth_dataset, batch_size=512, shuffle=False, num_workers=4)
    
    real_images_path = f"{args.dataset_root}temp_{args.method.lower()}{approach_suffix}_{args.dataset}/"
    os.makedirs(real_images_path, exist_ok=True)

    print(f"[LOG] Exporting {args.dataset} as PNGs for metrics computation")
    counter = 0
    total = len(test_Dataset)
    for batch in tqdm(real_loader, desc="Saving CIFAR-10 images"):
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        for img in images:
            save_image(img, os.path.join(real_images_path, f'{counter:06d}.png'))
            counter += 1
            if counter >= total:
                break
        if counter >= total:
            break

    ### -------------------------------- Metrics computation --------------------------------- ###
    formatter = TextFormatter()
    comp_str = formatter.format(
        f"Starting metrics computation",
        color = "blue", 
        style = ["bold"], 
        separator = True
    )
    print(comp_str)
    X_real, energy_real = extract_features(real_loader,args.gpu,teacher,"REAL")
    X_synth, energy_synth = extract_features(synth_loader,args.gpu,teacher,"SYNTHETIC")

    print("[LOG] Plot t-SNE")
    plot_tsne(X_real, X_synth, "tsne.png", args.dataset.upper(), args.method.upper() + " Synthetic samples")

    print("[LOG] Plot UMAP")
    plot_umap(X_real, X_synth, "umap.png", args.dataset.upper(), args.method.upper() + " Synthetic samples")

    print("[LOG] Plot Energy")
    plot_energy(energy_real, energy_synth, "energy.png", args.dataset.upper(), args.method.upper() + " Synthetic samples")

    # Combina le tre immagini in un'unica figura
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))  # 3 righe x 1 colonna
    for ax, path in zip(
        axes,
        ["tsne.png", "umap.png", "energy.png"]    
    ):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(filename_out)
    plt.close()

    # Rimuove i PNG temporanei
    for f in ["tsne.png", "umap.png", "energy.png"]:
        os.remove(f)
    
    shutil.rmtree(real_images_path)