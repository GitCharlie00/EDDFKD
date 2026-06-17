import argparse
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pytz
import re
import seaborn as sns
import shutil
import sys
import smtplib
import torch
import torch.nn as nn
import umap

from cleanfid import fid
from contextlib import contextmanager
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from itertools import cycle
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpecFromSubplotSpec
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.stats import entropy, wasserstein_distance
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_kernels
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

def send_email(network, dataset, best_acc=0, approach="", method="", error=False, error_msg="",teacher=""):
    rome_tz = pytz.timezone("Europe/Rome")
    current_time = datetime.now(rome_tz)
    print_date = current_time.strftime("%d/%m/%Y %H:%M:%S")

    YOUR_GOOGLE_EMAIL = "sapienza.amr.2024@gmail.com"  # The email you setup to send the email using app password
    YOUR_GOOGLE_EMAIL_APP_PASSWORD = "ytdo rxyn bocy ezap"  # The app password you generated

    # Setup the SMTP server
    smtpserver = smtplib.SMTP_SSL("smtp.gmail.com", 465)
    smtpserver.ehlo()
    smtpserver.login(YOUR_GOOGLE_EMAIL, YOUR_GOOGLE_EMAIL_APP_PASSWORD)

    # Create the email
    sent_from = YOUR_GOOGLE_EMAIL
    sent_to = sent_from  # Send it to self (as test)

    if not error:
        subject = "[RTX5000] Training ended"
        if approach != "":
            body = f"The KD {approach}{method} of {teacher.upper()}->{network.upper()} is correctly ended on {print_date}. The configuration was: \n\t- Dataset: {dataset.upper()} \n\t- Best accuracy: {best_acc} \n\t- Method:{method}"
        else:
            body = f"The train of {teacher.upper()}->{network.upper()} is correctly ended on {print_date}. The configuration was: \n\t- Dataset: {dataset.upper()} \n\t- Best accuracy: {best_acc}"
    else:
        subject = "!!! [WKS] Training ERROR !!!"
        body = f"The train of {teacher.upper()}->{network.upper()} is UNEXPLECTEDLY ended on {print_date}. The error was: \n\n {error_msg}"

    # Use MIMEMultipart to create an email with subject and body
    message = MIMEMultipart()
    message["From"] = sent_from
    message["To"] = sent_to
    message["Subject"] = subject

    # Attach the email body
    message.attach(MIMEText(body, "plain"))

    # Send the email
    smtpserver.sendmail(sent_from, sent_to, message.as_string())

    # Close the connection
    smtpserver.close()

class TextFormatter:
    # ANSI escape codes for colors
    COLORS = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "purple": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
    }

    # ANSI escape codes for text styles
    STYLES = {
        "normal": "\033[0m",
        "bold": "\033[1m",
        "underline": "\033[4m",
        "blink": "\033[5m",
    }

    def __init__(self):
        self.terminal_width = shutil.get_terminal_size().columns
        self.max_length = 100

    def format(self, text, color=None, style=None, separator=False):
        color_code = self.COLORS.get(color, "")
        style_code = "".join([self.STYLES.get(style, '') for style in style]) if style else ""
        formatted_text = f"{style_code}{color_code}{text}{self.STYLES['normal']}"

        if separator:
            separator_length = self.max_length - len(formatted_text) - 6
            separator_line = f"{self.STYLES['bold']}{self.COLORS.get(color, '')}{'*' * (separator_length // 2)}{self.STYLES['normal']}"
            formatted_text = f"{separator_line} {formatted_text} {separator_line}"

        return formatted_text

def hardware_check(mode,gpu_device=0):
    if mode == "train" or mode == "test" or mode == "dfkd" or mode == "plot":
        device = f"cuda:{int(gpu_device)}" if torch.cuda.is_available() else "cpu"
        print(device)
    else:
        device = "cpu"
    
    print(f"[LOG] Current device: {device}")
    if "cuda" in device:
        print("ci sono")
        prop = torch.cuda.get_device_properties(device)
        print("ci sono")
        print(f"[LOG] Device name: {prop.name}")
        print(f"[LOG] Device memory: {prop.total_memory/ (1024 ** 2)} MB")
        print(f"[LOG] Device processors: {prop.multi_processor_count}")
        
    return device

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

def init_arg_parser():
    parser = argparse.ArgumentParser(description="D-DFKD PyTorch implementation.", fromfile_prefix_chars="@")
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    # Program
    parser.add_argument("--mode",             type=str,  help="Train or test",    default="")
    parser.add_argument("--gpu_id",           type=int,  help="GPU index",        default="")

    # Dataset
    parser.add_argument("--dataset",          type=str,  help="Dataset to use",   default="")
    parser.add_argument("--ood_dataset",      type=str,  help="OOD Dataset",      default="")
    parser.add_argument("--dataset_root",     type=str,  help="Path to the data", default="work/data/ssd_datasets/")
    parser.add_argument("--num_classes",      type=int,  help="Classes number",   default=10)
    parser.add_argument("--ood_classes",      type=int,  help="OOD classes",      default=10)
    parser.add_argument("--num_workers",      type=int,  help="Parallel workers", default=6)

    # Results
    parser.add_argument("--save_path",        type=str,  help="Results path",     default="work/data/ssd_results/")
    parser.add_argument("--suffix",           type=str,  help="User suffix",      default="")
    parser.add_argument("--log_interval",     type=int,  help="Log interval",     default=10)
    parser.add_argument("--data_clear",       type=int,  help="Data clear",       default=10)
    parser.add_argument("--keep_last",        type=int,  help="Last images",      default=10)

    # Network
    parser.add_argument("--network",          type=str,    help="Network to train", default="")
    parser.add_argument("--t_network",        type=str,    help="DFKD teacher",     default="")
    parser.add_argument("--s_network",        type=str,    help="DFKD student",     default="")

    # DFKD
    parser.add_argument("--approach",         type=str,    help="DFKD method",      default="")
    parser.add_argument("--method",           type=str,    help="Method name",      default="")
    parser.add_argument('--kdci',                          help="KDCI on/off",      action='store_true')
    parser.add_argument("--confounder_size",  type=int,    help="Cluster N",        default=8)
    parser.add_argument("--hidden_dim",       type=int,    help="Attention dim",    default=256)
    
    # Energy terms
    parser.add_argument('--ood_loss',                      help="OOD loss on/off",  action='store_true')
    parser.add_argument('--additive_loss',                 help="Additive energy",  action='store_true')
    parser.add_argument('--gamma_ood',        type=float,  help="OOD gamma value",  default=0.5)
    parser.add_argument('--gamma_adaptive',   type=str,    help="Adaptive gamma",   default="single")
    parser.add_argument('--new_gamma_ood',    type=float,  help="OOD gamma value",  default=0.1)
    parser.add_argument('--in_energy_target', type=float,  help="OOD target value", default=0.1)
    parser.add_argument('--out_energy_target',type=float,  help="ID target value",  default=0.1)
    parser.add_argument('--g_reset',                       help="G reset",  action='store_true')
    parser.add_argument('--g_penality',                    help="G diversity",  action='store_true')
    parser.add_argument('--s_energy_match',                help="T-S match",    action='store_true')
    parser.add_argument('--energy_kd',                     help="Energy-weighted KD on/off", action='store_true')
    parser.add_argument('--energy_kd_beta',   type=float,  help="Energy-KD gate sharpness",  default=1.0)
    parser.add_argument('--energy_temp',                   help="Energy-adaptive KD temperature on/off", action='store_true')
    parser.add_argument('--energy_temp_base', type=float,  help="Base KD temperature tau_0", default=4.0)
    parser.add_argument('--energy_temp_alpha',type=float,  help="Energy-adaptive temp strength (0=global T)", default=1.0)

    # Training initialization
    parser.add_argument("--lr",               type=float,  help="Learning rate",   default=0.001)
    parser.add_argument("--momentum",         type=float,  help="Momentum",        default=0.9)
    parser.add_argument("--weight_decay",     type=float,  help="Weight decay",    default=5e-4)

    # Training parameters
    parser.add_argument("--train_batch_size", type=int,    help="Train batch size", default=4)
    parser.add_argument("--epochs",           type=int,    help="Number of epochs", default=2)

    # Generator losses - DAFL
    parser.add_argument("--act",              type=float,  help="Loss act",        default=0.1)
    parser.add_argument("--ie",               type=float,  help="Loss ie"  ,       default=5.0)
    
    # Generator losses - FAST
    parser.add_argument("--oh",               type=float,  help="Loss oh",         default=5.0)
    parser.add_argument("--adv",              type=float,  help="Loss adv",        default=0.1)
    parser.add_argument("--feat",             type=float,  help="Loss feat",       default=5.0)
    parser.add_argument("--bn_mmt",           type=float,  help="Hook values",     default=0.9)
    parser.add_argument("--warmup",           type=int,    help="G warmup",        default=1)
    parser.add_argument("--lr_z",             type=float,  help="Embedding lr",    default=0.1)
    
    # Data-free approaches parameters
    parser.add_argument("--z_dim",            type=int,    help="Latent dim",      default=100)
    parser.add_argument("--lr_G",             type=float,  help="Generator lr",    default=0.0002)
    parser.add_argument("--lr_S",             type=float,  help="Student lr",      default=0.1)
    parser.add_argument("--temperature",      type=float,  help="KL temperature",  default=20)
    parser.add_argument("--milestone_1",      type=int,    help="Lr milestone 1",  default=800) 
    parser.add_argument("--milestone_2",      type=int,    help="Lr milestone 2",  default=1200) 
    parser.add_argument("--dataset_size",     type=int,    help="Dataset size",    default=50000)
    parser.add_argument('--feature_extr',                  help="Extr layer",      action='store_true')
    parser.add_argument("--kd_steps",         type=int,    help="KD steps",        default=1200)
    parser.add_argument("--ep_steps",         type=int,    help="Epoch steps",     default=1200)
    parser.add_argument("--g_steps",          type=int,    help="Generator steps", default=1200)

    # CMI parameters
    parser.add_argument("--cr",               type=float,  help="Cr param",        default=0.1)
    parser.add_argument("--cr_T",             type=float,  help="Cr_t param",      default=5.0)

    # DEEPINV parameters
    parser.add_argument("--tv",               type=float,  help="TV param",        default=5.0)

    # MOSAICK parameters
    parser.add_argument("--align",            type=float,  help="align",           default=0.1)
    parser.add_argument("--local",            type=float,  help="local",           default=0.1)
    parser.add_argument("--nc",               type=int,    help="nc",              default=1)
    parser.add_argument("--img_size",         type=int,    help="img_size",        default=1)
    parser.add_argument("--ndf",              type=int,    help="ndf",             default=1)
    
    # DFND parameters
    parser.add_argument("--lr_N",             type=float,  help="Noise lr",        default=0.0002)
    parser.add_argument("--num_select",       type=int,    help="Num select",      default=100000)

    # Testing parameters
    parser.add_argument("--test_batch_size",  type=int,    help="Test batch size", default=1)

    # Plotting
    parser.add_argument('--calc_target',                   help="Energy target",   action='store_true')
                                                                            
    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = "@" + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    return args

def print_args(args):
    for key, value in args.__dict__.items():
        print(f"[LOG] {key}: {value}")

def print_examples(args, train_loader, test_loader):
    """
    Stampa/salva due griglie di immagini:
      - una per il primo batch di TRAIN
      - una per il primo batch di TEST

    Ora il dataset restituisce (img_tensor, numeric_label, text_label),
    quindi mostriamo sia l"intero sia la stringa della classe sopra ogni immagine.
    """

    # → Ogni elemento di train_loader/test_loader è: (batch_tensor, batch_numeric_labels, batch_text_labels)
    train_batch, train_numeric_labels, train_text_labels = next(iter(train_loader))
    test_batch,  test_numeric_labels,  test_text_labels  = next(iter(test_loader))

    # Quante immagini vogliamo mostrare (fino a 8)
    N = min(8, train_batch.size(0))
    M = min(8, test_batch.size(0))

    # Valori di mean/std usati nella normalizzazione del dataset
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)

    # --------------------------------------------------------
    # 1) Creiamo/salviamo la griglia per il batch di TRAIN
    # --------------------------------------------------------
    # Calcoliamo dinamicamente righe e colonne per N immagini (“quasi quadrata”)
    ncol_train = math.ceil(math.sqrt(N))
    nrow_train = math.ceil(N / ncol_train)

    # Ogni miniatura 3×3 pollici (regolabile a piacere)
    fig1 = plt.figure(figsize=(3 * ncol_train, 3 * nrow_train))

    for i in range(N):
        ax = fig1.add_subplot(nrow_train, ncol_train, i + 1)

        # Prendiamo l"immagine normalizzata e facciamo un-normalization
        img = train_batch[i].cpu()
        img_unnorm = img * std + mean                # tornare in [0,1]
        img_unnorm = torch.clamp(img_unnorm, 0.0, 1.0)
        img_np = img_unnorm.permute(1, 2, 0).numpy()  # shape (H, W, 3)
        
        ax.imshow(img_np)
        ax.axis("off")

        # Titolo sopra l’immagine: “<numeric_label> – <testo_classe>”
        num_lbl = train_numeric_labels[i].item()
        txt_lbl = train_text_labels[i]
        ax.set_title(f"{num_lbl} - {txt_lbl}", fontsize=10, fontweight="bold")

    # Titolo generale in alto (grassetto, font size più grande)
    fig1.suptitle("Train Images", fontsize=16, fontweight="bold")
    fig1.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(args.save_path, exist_ok=True)
    train_output = os.path.join(args.save_path, "train_batch_grid.pdf")
    plt.savefig(train_output)
    plt.close(fig1)

    # --------------------------------------------------------
    # 2) Creiamo/salviamo la griglia per il batch di TEST
    # --------------------------------------------------------
    ncol_test = math.ceil(math.sqrt(M))
    nrow_test = math.ceil(M / ncol_test)

    fig2 = plt.figure(figsize=(3 * ncol_test, 3 * nrow_test))

    for i in range(M):
        ax = fig2.add_subplot(nrow_test, ncol_test, i + 1)

        img = test_batch[i].cpu()
        img_unnorm = img * std + mean
        img_unnorm = torch.clamp(img_unnorm, 0.0, 1.0)
        img_np = img_unnorm.permute(1, 2, 0).numpy()
        
        ax.imshow(img_np)
        ax.axis("off")

        num_lbl = test_numeric_labels[i].item()
        txt_lbl = test_text_labels[i]
        ax.set_title(f"{num_lbl} - {txt_lbl}", fontsize=10, fontweight="bold")

    fig2.suptitle("Test Images", fontsize=16, fontweight="bold")
    fig2.tight_layout(rect=[0, 0, 1, 0.95])

    test_output = os.path.join(args.save_path, "test_batch_grid.pdf")
    plt.savefig(test_output)
    plt.close(fig2)

def approach_label(args):
    parts = []
    
    if args.ood_loss:
        prefix = "add_" if args.additive_loss else ""
        
        # if args.gamma_adaptive != "none":
        #     old_g = str(args.gamma_ood) 
        #     new_g = str(args.new_gamma_ood) 
        #     ood_part = f"{prefix}ood_{args.gamma_adaptive}_{old_g}_to_{new_g}"
        # else:
        #     ood_part = f"{prefix}ood_{str(args.gamma_ood)}"
        
        ood_part = f"{prefix}ood_{str(args.gamma_ood)}"    
        parts.append(ood_part)

    if args.kdci:
        parts.append("kdci")

    if getattr(args, "energy_kd", False):
        parts.append("ekd")

    if getattr(args, "energy_temp", False):
        parts.append("etemp")

    # if args.g_reset:
        # parts.append("g_reset")
    
    # if args.g_penality:
        # parts.append("g_penality")
    
    # if args.s_energy_match:
        # parts.append("match")
        
    if not parts:
        return ""
        
    return f"_{'_'.join(parts)}"
    

def generate_training_report_pdf(args, history, t_model_name, dataset, best_acc):
    """
    Genera un PDF con i grafici di loss e accuracy del training.
    
    Args:
        history (dict): Dizionario contenente train_loss, val_loss, train_accuracy, val_accuracy
        args: Oggetto argomenti con parametri di training
        t_model_name (str): Nome del modello teacher
        best_acc (float): Migliore accuracy raggiunta
        save_path (str): Percorso dove salvare il PDF
    """
    
    # Nome del file PDF
    pdf_filename = f"{args.save_path}{args.dataset}/{t_model_name.lower()}_{dataset.lower()}_training_report.pdf"
    
    # Calcola metriche per epoca (assumendo stesso numero di batch per epoca)
    train_batches_per_epoch = len(history["train_loss"]) // args.epochs
    val_batches_per_epoch = len(history["val_loss"]) // args.epochs
    
    # Raggruppa per epoche
    train_loss_epochs = []
    train_acc_epochs = []
    val_loss_epochs = []
    val_acc_epochs = []
    
    for epoch in range(args.epochs):
        # Training metrics per epoca
        start_idx = epoch * train_batches_per_epoch
        end_idx = (epoch + 1) * train_batches_per_epoch
        train_loss_epochs.append(np.mean(history["train_loss"][start_idx:end_idx]))
        train_acc_epochs.append(np.mean(history["train_accuracy"][start_idx:end_idx]))
        
        # Validation metrics per epoca
        start_idx = epoch * val_batches_per_epoch
        end_idx = (epoch + 1) * val_batches_per_epoch
        val_loss_epochs.append(np.mean(history["val_loss"][start_idx:end_idx]))
        val_acc_epochs.append(np.mean(history["val_accuracy"][start_idx:end_idx]))
    
    epochs_range = range(1, args.epochs + 1)
    
    with PdfPages(pdf_filename) as pdf:
        # Configurazione stile moderno con sfondo bianco
        plt.style.use('default')
        colors = {
            'train': '#2E86AB',      # Blu professionale
            'val': '#F24236',        # Rosso professionale
            'bg': '#FFFFFF',         # Bianco
            'grid': '#E0E0E0',       # Grigio chiaro per griglia
            'text': '#2C3E50'        # Grigio scuro per testo
        }
        
        # Pagina 1: Tutti i grafici in una pagina
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor(colors['bg'])
        
        # Converti batch in epoche per i grafici batch-wise
        train_batch_epochs = np.linspace(1, args.epochs, len(history["train_loss"]))
        val_batch_epochs = np.linspace(1, args.epochs, len(history["val_loss"]))
        train_acc_batch_epochs = np.linspace(1, args.epochs, len(history["train_accuracy"]))
        val_acc_batch_epochs = np.linspace(1, args.epochs, len(history["val_accuracy"]))
        
        # Training Loss ibrido
        # Background: andamento per batch (più chiaro)
        ax1.plot(train_batch_epochs, history["train_loss"], color=colors['train'], alpha=0.3, linewidth=1, label='Per Batch')
        # Foreground: smoothing per epoca (più visibile)
        ax1.plot(epochs_range, train_loss_epochs, color=colors['train'], 
                linewidth=3, marker='o', markersize=3, alpha=0.9, label='Per Epoch')
        ax1.set_title('Training Loss', fontsize=16, fontweight='bold', color=colors['text'], pad=15)
        ax1.set_xlabel('Epochs', fontsize=12, color=colors['text'])
        ax1.set_xticks(range(1, args.epochs + 1, 10))
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_ylabel('Loss', fontsize=12, color=colors['text'])
        ax1.grid(True, alpha=0.3, color=colors['grid'])
        ax1.set_facecolor(colors['bg'])
        ax1.tick_params(colors=colors['text'])
        ax1.legend(frameon=True, fontsize=10)
        
        # Validation Loss ibrido
        # Background: andamento per batch (più chiaro)
        ax2.plot(val_batch_epochs, history["val_loss"], color=colors['val'], alpha=0.3, linewidth=1, label='Per Batch')
        # Foreground: smoothing per epoca (più visibile)
        ax2.plot(epochs_range, val_loss_epochs, color=colors['val'], linewidth=3, marker='s', markersize=3, alpha=0.9, label='Per Epoch')
        ax2.set_title('Validation Loss', fontsize=16, fontweight='bold', color=colors['text'], pad=15)
        ax2.set_xlabel('Epochs', fontsize=12, color=colors['text'])
        ax2.set_xticks(range(1, args.epochs + 1, 10))
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylabel('Loss', fontsize=12, color=colors['text'])
        ax2.grid(True, alpha=0.3, color=colors['grid'])
        ax2.set_facecolor(colors['bg'])
        ax2.tick_params(colors=colors['text'])
        ax2.legend(frameon=True, fontsize=10)
        
        # Training Accuracy ibrido
        # Background: andamento per batch (più chiaro)
        ax3.plot(train_acc_batch_epochs, history["train_accuracy"], color=colors['train'], 
                alpha=0.3, linewidth=1, label='Per Batch')
        # Foreground: smoothing per epoca (più visibile)
        ax3.plot(epochs_range, train_acc_epochs, color=colors['train'], linewidth=3, marker='o', markersize=3, alpha=0.9, label='Per Epoch')
        ax3.set_title('Training Accuracy', fontsize=16, fontweight='bold', color=colors['text'], pad=15)
        ax3.set_xlabel('Epochs', fontsize=12, color=colors['text'])
        ax3.set_xticks(range(1, args.epochs + 1, 10))
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_ylabel('Accuracy', fontsize=12, color=colors['text'])
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3, color=colors['grid'])
        ax3.set_facecolor(colors['bg'])
        ax3.tick_params(colors=colors['text'])
        ax3.legend(frameon=True, fontsize=10)
        
        # Validation Accuracy ibrido
        # Background: andamento per batch (più chiaro)
        ax4.plot(val_acc_batch_epochs, history["val_accuracy"], color=colors['val'], alpha=0.3, linewidth=1, label='Per Batch')
        # Foreground: smoothing per epoca (più visibile)
        ax4.plot(epochs_range, val_acc_epochs, color=colors['val'], linewidth=3, marker='s', markersize=3, alpha=0.9, label='Per Epoch')
        ax4.set_title('Validation Accuracy', fontsize=16, fontweight='bold', color=colors['text'], pad=15)
        ax4.set_xlabel('Epochs', fontsize=12, color=colors['text'])
        ax4.set_xticks(range(1, args.epochs + 1, 10))
        ax4.tick_params(axis='x', rotation=45)
        ax4.set_ylabel('Accuracy', fontsize=12, color=colors['text'])
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3, color=colors['grid'])
        ax4.set_facecolor(colors['bg'])
        ax4.tick_params(colors=colors['text'])
        ax4.legend(frameon=True, fontsize=10)
        
        plt.suptitle(f'{t_model_name} - Training Metrics', fontsize=20, fontweight='bold', color=colors['text'], y=0.99)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', facecolor=colors['bg'])
        plt.close()
        
        # Pagina 2: Training Report ordinato e preciso
        fig, ax = plt.subplots(figsize=(12, 16))
        fig.patch.set_facecolor(colors['bg'])
        ax.axis('off')
        
        # Calcola statistiche finali
        final_train_loss = train_loss_epochs[-1]
        final_val_loss = val_loss_epochs[-1]
        final_train_acc = train_acc_epochs[-1]
        final_val_acc = val_acc_epochs[-1]
        
        # Layout ordinato e preciso
        y_start = 0.95
        line_height = 0.035
        section_gap = 0.05
        
        current_y = y_start
        
        # Titolo principale
        ax.text(0.5, current_y, 'TRAINING REPORT', fontsize=24, fontweight='bold', 
               color=colors['train'], ha='center', va='top')
        current_y -= 0.08
        
        # Linea separatrice
        ax.plot([0.1, 0.9], [current_y, current_y], color=colors['train'], linewidth=2)
        current_y -= section_gap
        
        # Sezione Model Information
        ax.text(0.1, current_y, 'MODEL INFORMATION', fontsize=16, fontweight='bold', 
               color=colors['text'])
        current_y -= line_height
        
        model_info = [
            f'Model Name: {t_model_name}',
            f'Dataset: {args.dataset}',
            f'Number of Classes: {args.num_classes}'
        ]
        
        for info in model_info:
            ax.text(0.15, current_y, info, fontsize=12, color=colors['text'])
            current_y -= line_height
        
        current_y -= section_gap
        
        # Sezione Training Configuration
        ax.text(0.1, current_y, 'TRAINING CONFIGURATION', fontsize=16, fontweight='bold', 
               color=colors['text'])
        current_y -= line_height
        
        training_config = [
            f'Total Epochs: {args.epochs}',
            f'Train Batch Size: {args.train_batch_size}',
            f'Test Batch Size: {args.test_batch_size}',
            f'Initial Learning Rate: {args.lr}',
            f'Momentum: {args.momentum}',
            f'Weight Decay: {args.weight_decay}',
            f'Optimizer: SGD',
            f'Loss Function: CrossEntropyLoss'
        ]
        
        for config in training_config:
            ax.text(0.15, current_y, config, fontsize=12, color=colors['text'])
            current_y -= line_height
        
        current_y -= section_gap
        
        # Sezione Training Results
        ax.text(0.1, current_y, 'TRAINING RESULTS', fontsize=16, fontweight='bold', 
               color=colors['text'])
        current_y -= line_height
        
        results = [
            f'Best Validation Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)',
            f'Final Training Loss: {final_train_loss:.4f}',
            f'Final Validation Loss: {final_val_loss:.4f}',
            f'Final Training Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)',
            f'Final Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)'
        ]
        
        for result in results:
            ax.text(0.15, current_y, result, fontsize=12, color=colors['text'])
            current_y -= line_height
        
        current_y -= section_gap
        
        # Sezione Statistics
        ax.text(0.1, current_y, 'STATISTICS', fontsize=16, fontweight='bold', 
               color=colors['text'])
        current_y -= line_height
        
        statistics = [
            f'Total Training Batches: {len(history["train_loss"])}',
            f'Total Validation Batches: {len(history["val_loss"])}',
            f'Training Batches per Epoch: {train_batches_per_epoch}',
            f'Validation Batches per Epoch: {val_batches_per_epoch}',
            '\n',
            '\n'
        ]
        
        for stat in statistics:
            ax.text(0.15, current_y, stat, fontsize=12, color=colors['text'])
            current_y -= line_height
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        pdf.savefig(fig, bbox_inches='tight', facecolor=colors['bg'])
        plt.close()
    
    print(f"[LOG] Training report PDF saved: {pdf_filename}")


def generate_test_report_pdf(args, metrics, model_name, dataset, class_names, t_model_name="", approach_suffix="", student_test=False):
    """
    Genera un PDF con tutti i risultati del testing.
    
    Args:
        args: Oggetto argomenti con parametri di testing
        metrics (dict): Dizionario contenente tutte le metriche calcolate
        model_name (str): Nome del modello
        dataset (str): Nome del dataset
        class_names (list): Lista dei nomi delle classi
    """
    
    # Nome del file PDF
    if student_test:
        pdf_filename = f"{args.save_path}{dataset}/{t_model_name.lower()}_{model_name.lower()}_{dataset.lower()}_{args.approach.lower()}_{args.method.lower()}{approach_suffix}_test_report.pdf"
    else:
        pdf_filename = f"{args.save_path}{dataset}/{model_name.lower()}_{dataset.lower()}_test_report.pdf"
    
    with PdfPages(pdf_filename) as pdf:
        # Configurazione stile moderno
        plt.style.use('default')
        colors = {
            'primary': '#2E86AB',      # Blu professionale
            'secondary': '#F24236',    # Rosso professionale
            'accent': '#A23B72',       # Viola
            'success': '#F18F01',      # Arancione
            'bg': '#FFFFFF',           # Bianco
            'grid': '#E0E0E0',         # Grigio chiaro per griglia
            'text': '#2C3E50'          # Grigio scuro per testo
        }
        
        # Pagina 1: Test Report Summary
        fig, ax = plt.subplots(figsize=(12, 16))
        fig.patch.set_facecolor(colors['bg'])
        ax.axis('off')
        
        # Layout ordinato
        y_start = 0.95
        line_height = 0.035
        section_gap = 0.05
        current_y = y_start
        
        # Titolo principale
        ax.text(0.5, current_y, 'TEST REPORT', fontsize=24, fontweight='bold', 
               color=colors['primary'], ha='center', va='top')
        current_y -= 0.08
        
        # Linea separatrice
        ax.plot([0.1, 0.9], [current_y, current_y], color=colors['primary'], linewidth=2)
        current_y -= section_gap
        
        # Sezione Model Information
        ax.text(0.1, current_y, 'MODEL INFORMATION', fontsize=16, fontweight='bold', 
               color=colors['text'])
        current_y -= line_height
        
        model_info = [
            f'Model Name: {model_name}',
            f'Dataset: {dataset}',
            f'Number of Classes: {len(class_names)}',
            f'Trained Epoch: {args.epochs}'
        ]
        
        for info in model_info:
            ax.text(0.15, current_y, info, fontsize=12, color=colors['text'])
            current_y -= line_height
        
        current_y -= section_gap
        
        # Sezione Overall Performance
        ax.text(0.1, current_y, 'OVERALL PERFORMANCE', fontsize=16, fontweight='bold', 
               color=colors['text'])
        current_y -= line_height
        
        performance = [
            f'Test Accuracy: {metrics["test_accuracy"]:.4f} ({metrics["test_accuracy"]*100:.2f}%)',
            f'Average Test Loss: {metrics["avg_test_loss"]:.4f}',
            f'Test Time: {metrics["test_time"]:.2f} seconds',
            f'Samples per Second: {len(metrics["confusion_matrix"].ravel()) / metrics["test_time"]:.2f}'
        ]
        
        for perf in performance:
            ax.text(0.15, current_y, perf, fontsize=12, color=colors['text'])
            current_y -= line_height
        
        current_y -= section_gap
        
        # Sezione Macro Averages
        ax.text(0.1, current_y, 'MACRO AVERAGES', fontsize=16, fontweight='bold', 
               color=colors['text'])
        current_y -= line_height
        
        macro_metrics = [
            f'Precision (Macro): {metrics["precision_macro"]:.4f}',
            f'Recall (Macro): {metrics["recall_macro"]:.4f}',
            f'F1-Score (Macro): {metrics["f1_macro"]:.4f}',
            f'F2-Score (Macro): {metrics["f2_macro"]:.4f}'
        ]
        
        for metric in macro_metrics:
            ax.text(0.15, current_y, metric, fontsize=12, color=colors['text'])
            current_y -= line_height
        
        current_y -= section_gap
        
        # Sezione Weighted Averages
        ax.text(0.1, current_y, 'WEIGHTED AVERAGES', fontsize=16, fontweight='bold', 
               color=colors['text'])
        current_y -= line_height
        
        weighted_metrics = [
            f'Precision (Weighted): {metrics["precision_weighted"]:.4f}',
            f'Recall (Weighted): {metrics["recall_weighted"]:.4f}',
            f'F1-Score (Weighted): {metrics["f1_weighted"]:.4f}',
            f'F2-Score (Weighted): {metrics["f2_weighted"]:.4f}'
        ]
        
        for metric in weighted_metrics:
            ax.text(0.15, current_y, metric, fontsize=12, color=colors['text'])
            current_y -= line_height
        
        current_y -= section_gap
        
        # Sezione ROC AUC
        ax.text(0.1, current_y, 'ROC AUC SCORES', fontsize=16, fontweight='bold', 
               color=colors['text'])
        current_y -= line_height
        
        if len(class_names) > 2:
            roc_info = [
                f'Micro-Average AUC: {metrics["roc_auc"]["micro"]:.4f}',
                f'Macro-Average AUC: {metrics["roc_auc"]["macro"]:.4f}',
                '\n',
                '\n'
            ]
            
            for info in roc_info:
                ax.text(0.15, current_y, info, fontsize=12, color=colors['text'])
                current_y -= line_height
        else:
            ax.text(0.15, current_y, f'Binary AUC: {metrics["roc_auc_bin"]:.4f}', 
                   fontsize=12, color=colors['text'])
            current_y -= line_height
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        pdf.savefig(fig, bbox_inches='tight', facecolor=colors['bg'])
        plt.close()
        
        # Pagina 2: Confusion Matrix
        if len(class_names) == 100:  # CIFAR-100 case - improved readability
            fig, ax = plt.subplots(figsize=(16, 14))
            fig.patch.set_facecolor(colors['bg'])
            
            # Normalizza confusion matrix per percentuali
            cm_normalized = metrics['confusion_matrix'].astype('float') / metrics['confusion_matrix'].sum(axis=1)[:, np.newaxis] * 100
            
            # Crea heatmap con colormap ottimizzata per CIFAR-100
            im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues', aspect='auto')
            
            # Colorbar più piccola e posizionata meglio
            cbar = ax.figure.colorbar(im, ax=ax, label='Percentage (%)', shrink=0.8, pad=0.02)
            cbar.ax.tick_params(labelsize=10)
            
            # Non aggiungere testo nelle celle per CIFAR-100 (troppo piccole)
            # Solo per celle con valori molto alti (>50%) per evidenziare predizioni corrette
            thresh = 50.0  # Soglia più alta per CIFAR-100
            for i in range(0, cm_normalized.shape[0], 5):  # Solo ogni 5a classe
                for j in range(0, cm_normalized.shape[1], 5):
                    if cm_normalized[i, j] > thresh:
                        ax.text(j, i, f'{cm_normalized[i, j]:.0f}%',
                               ha="center", va="center",
                               color="white", fontsize=6, fontweight='bold')
            
            # Personalizza assi per CIFAR-100
            tick_marks = np.arange(len(class_names))
            ax.set_xticks(tick_marks[::5])  # Ogni 5a classe
            ax.set_yticks(tick_marks[::5])
            
            # Labels più leggibili per CIFAR-100
            ax.set_xticklabels([f'{i}' for i in range(0, len(class_names), 5)], 
                              rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels([f'{i}' for i in range(0, len(class_names), 5)], 
                              fontsize=8)
            
            # Aggiungi griglia per migliorare la leggibilità
            ax.set_xticks(np.arange(len(class_names)) - 0.5, minor=True)
            ax.set_yticks(np.arange(len(class_names)) - 0.5, minor=True)
            ax.grid(which="minor", color="white", linestyle='-', linewidth=0.5, alpha=0.3)
            
            ax.set_ylabel('True Label (Class Index)', fontsize=12, color=colors['text'])
            ax.set_xlabel('Predicted Label (Class Index)', fontsize=12, color=colors['text'])
            ax.set_title('Confusion Matrix (Normalized) - CIFAR-100\nShowing every 5th class index', 
                        fontsize=14, fontweight='bold', color=colors['text'], pad=20)
            
        else:  # Altri dataset con meno classi
            fig, ax = plt.subplots(figsize=(12, 10))
            fig.patch.set_facecolor(colors['bg'])
            
            # Normalizza confusion matrix per percentuali
            cm_normalized = metrics['confusion_matrix'].astype('float') / metrics['confusion_matrix'].sum(axis=1)[:, np.newaxis] * 100
            
            # Crea heatmap
            im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
            ax.figure.colorbar(im, ax=ax, label='Percentage (%)')
            
            # Aggiungi testo nelle celle
            thresh = cm_normalized.max() / 2.
            for i in range(cm_normalized.shape[0]):
                for j in range(cm_normalized.shape[1]):
                    ax.text(j, i, f'{cm_normalized[i, j]:.1f}%\n({metrics["confusion_matrix"][i, j]})',
                           ha="center", va="center",
                           color="white" if cm_normalized[i, j] > thresh else "black",
                           fontsize=8 if len(class_names) <= 10 else 6)
            
            # Personalizza assi
            tick_marks = np.arange(len(class_names))
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(class_names, rotation=45, ha="right")
            ax.set_yticklabels(class_names)
            
            ax.set_ylabel('True Label', fontsize=12, color=colors['text'])
            ax.set_xlabel('Predicted Label', fontsize=12, color=colors['text'])
            ax.set_title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold', 
                        color=colors['text'], pad=20)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', facecolor=colors['bg'])
        plt.close()
        
        # Pagina 3: Per-Class Metrics
        if len(class_names) <= 20:  # Solo se non troppe classi
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.patch.set_facecolor(colors['bg'])
            
            x_pos = np.arange(len(class_names))
            
            # Precision per classe
            bars1 = ax1.bar(x_pos, metrics['precision'], color=colors['primary'], alpha=0.8)
            ax1.set_title('Precision per Class', fontsize=14, fontweight='bold', color=colors['text'])
            ax1.set_ylabel('Precision', fontsize=12, color=colors['text'])
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(class_names, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3, color=colors['grid'])
            ax1.set_ylim(0, 1)
            
            # Recall per classe
            bars2 = ax2.bar(x_pos, metrics['recall'], color=colors['secondary'], alpha=0.8)
            ax2.set_title('Recall per Class', fontsize=14, fontweight='bold', color=colors['text'])
            ax2.set_ylabel('Recall', fontsize=12, color=colors['text'])
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(class_names, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3, color=colors['grid'])
            ax2.set_ylim(0, 1)
            
            # F1-Score per classe
            bars3 = ax3.bar(x_pos, metrics['f1_score'], color=colors['accent'], alpha=0.8)
            ax3.set_title('F1-Score per Class', fontsize=14, fontweight='bold', color=colors['text'])
            ax3.set_ylabel('F1-Score', fontsize=12, color=colors['text'])
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(class_names, rotation=45, ha='right')
            ax3.grid(True, alpha=0.3, color=colors['grid'])
            ax3.set_ylim(0, 1)
            
            # Support per classe
            bars4 = ax4.bar(x_pos, metrics['support'], color=colors['success'], alpha=0.8)
            ax4.set_title('Support per Class', fontsize=14, fontweight='bold', color=colors['text'])
            ax4.set_ylabel('Number of Samples', fontsize=12, color=colors['text'])
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(class_names, rotation=45, ha='right')
            ax4.grid(True, alpha=0.3, color=colors['grid'])
            
            plt.suptitle(f'{model_name} - Per-Class Metrics', fontsize=20, fontweight='bold', 
                        color=colors['text'], y=0.98)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight', facecolor=colors['bg'])
            plt.close()
        
        # Pagina 4: ROC Curves (solo micro e macro averages)
        if len(class_names) > 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            fig.patch.set_facecolor(colors['bg'])
            
            # Plot solo micro-average e macro-average ROC curves
            ax.plot(metrics['fpr']['micro'], metrics['tpr']['micro'],
                   color=colors['primary'], linestyle='-', linewidth=4,
                   label=f'Micro-avg (AUC = {metrics["roc_auc"]["micro"]:.3f})')
            
            ax.plot(metrics['fpr']['macro'], metrics['tpr']['macro'],
                   color=colors['secondary'], linestyle='-', linewidth=4,
                   label=f'Macro-avg (AUC = {metrics["roc_auc"]["macro"]:.3f})')
            
            # Plot diagonal line
            ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Random Classifier')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=12, color=colors['text'])
            ax.set_ylabel('True Positive Rate', fontsize=12, color=colors['text'])
            ax.set_title('ROC Curves - Average Performance', fontsize=16, fontweight='bold', color=colors['text'])
            ax.legend(loc="lower right", fontsize=12)
            ax.grid(True, alpha=0.3, color=colors['grid'])
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight', facecolor=colors['bg'])
            plt.close()
        else:
            # Binary classification ROC curve
            fig, ax = plt.subplots(figsize=(10, 8))
            fig.patch.set_facecolor(colors['bg'])
            
            ax.plot(metrics['fpr_bin'], metrics['tpr_bin'], color=colors['primary'], lw=3,
                   label=f'ROC Curve (AUC = {metrics["roc_auc_bin"]:.3f})')
            ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Random Classifier')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=12, color=colors['text'])
            ax.set_ylabel('True Positive Rate', fontsize=12, color=colors['text'])
            ax.set_title('ROC Curve', fontsize=16, fontweight='bold', color=colors['text'])
            ax.legend(loc="lower right", fontsize=12)
            ax.grid(True, alpha=0.3, color=colors['grid'])
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight', facecolor=colors['bg'])
            plt.close()
    
    print(f"[LOG] Test report PDF saved: {pdf_filename}")

def generate_gen_training_report_pdf(args, history, t_model_name, s_model_name, dataset, best_acc, sample_images, label_count, approach_suffix):
    """
    Genera un PDF con i grafici di loss e accuracy del training DAFL.
    
    Args:
        history (dict): Dizionario contenente G_tot_train_loss, G_Loh_train_loss, G_Lat_train_loss, 
                       G_Lie_train_loss, S_tot_train_loss, S_train_accuracy, S_val_loss, S_val_accuracy
        args: Oggetto argomenti con parametri di training
        t_model_name (str): Nome del modello teacher
        s_model_name (str): Nome del modello student
        dataset (str): Nome del dataset
        best_acc (float): Migliore accuracy raggiunta
        sample_images (list): Lista di tuple (image_path, pseudo_label) per le immagini campione
    """
    
    # Nome del file PDF
    pdf_filename = f"{args.save_path}{dataset}/{t_model_name.lower()}_{s_model_name.lower()}_{dataset.lower()}_{args.approach.lower()}_{args.method.lower()}{approach_suffix}_training_report.pdf"
    
    # Calcola il numero di step per epoca
    steps_per_epoch = args.ep_steps // args.kd_steps
    
    # Raggruppa le metriche per epoche
    def group_by_epochs(data_list, steps_per_epoch):
        epochs_data = []
        for epoch in range(args.epochs):
            start_idx = epoch * steps_per_epoch
            end_idx = min((epoch + 1) * steps_per_epoch, len(data_list))
            if start_idx < len(data_list):
                epoch_data = data_list[start_idx:end_idx]
                if epoch_data:  # Solo se ci sono dati per questa epoca
                    epochs_data.append(np.mean(epoch_data))
                else:
                    epochs_data.append(0)  # Valore di default se non ci sono dati
            else:
                epochs_data.append(0)
        return epochs_data
    
    # Seleziona tutte le loss specifiche
    last_spec_loss_n = 8 if args.method == "dafl" or args.method == "fast" else 9

    G_train_loss = history["G_train_loss"]
    S_train_loss = history["S_train_loss"]
    S_train_accuracy = history["S_train_accuracy"]
    S_val_loss = history["S_val_loss"]
    S_val_accuracy = history["S_val_accuracy"]        
    G_spec_losses = [history[key] for key in list(history.keys())[5:last_spec_loss_n]]

    # Raggruppa tutte le metriche per epoche
    G_tot_loss_epochs = group_by_epochs(G_train_loss, steps_per_epoch)
    S_train_loss_epochs = group_by_epochs(S_train_loss, steps_per_epoch)
    S_train_acc_epochs = group_by_epochs(S_train_accuracy, steps_per_epoch) if "S_train_accuracy" in history and len(S_train_accuracy) > 0 else []
    G_spec_losses_epochs = []
    for spec_loss in G_spec_losses:
        G_spec_losses_epochs.append(group_by_epochs(spec_loss["values"], steps_per_epoch))
    
    # Per validation loss e accuracy, raggruppa per numero di batch di validazione per epoca
    # Assumiamo che ogni epoca abbia lo stesso numero di batch di validazione
    val_batches_per_epoch = len(S_val_loss) // args.epochs if "S_val_loss" in history and len(S_val_loss) > 0 else 1
    S_val_loss_epochs = group_by_epochs(S_val_loss, val_batches_per_epoch) if "S_val_loss" in history and len(S_val_loss) > 0 else []
    S_val_acc_epochs = group_by_epochs(S_val_accuracy, val_batches_per_epoch) if "S_val_accuracy" in history and len(S_val_accuracy) > 0 else []
    
    generation_losses = [history[key] for key in list(history.keys())[last_spec_loss_n:last_spec_loss_n+4]]
    generation_losses_epochs = []
    for gen_loss_item in generation_losses:
        generation_losses_epochs.append(group_by_epochs(gen_loss_item["values"],steps_per_epoch))

    distillation_losses = [history[key] for key in list(history.keys())[last_spec_loss_n+4:last_spec_loss_n+9]]
    distillation_losses_epochs = []
    for dist_loss_item in distillation_losses:
        distillation_losses_epochs.append(group_by_epochs(dist_loss_item["values"],steps_per_epoch))

    epochs_range = range(1, args.epochs + 1)
    
    with PdfPages(pdf_filename) as pdf:
        # Configurazione stile moderno con sfondo bianco
        plt.style.use('default')
        colors = {
            'generator': '#E74C3C',     # Rosso per Generator
            'student': '#3498DB',       # Blu per Student
            'teacher': '#2ECC71',       # Verde per Teacher
            'bg': '#FFFFFF',            # Bianco
            'grid': '#E0E0E0',          # Grigio chiaro per griglia
            'text': '#2C3E50',          # Grigio scuro per testo
            'train': '#3498DB',         # Blu per training
            'val': '#E74C3C',           # Rosso per validation
            'val_comp': '#F4C2C2'       # Rosso per validation comparison con train
        }

        spec_loss_colors = ['#F39C12','#9B59B6','#1ABC9C','#F1C40F']
        for idx in range(len(G_spec_losses)):
            name = "L" + str(idx+1)
            colors[name] = spec_loss_colors[idx]

        colors_hex = [
            "#1f77b4",  # blu
            "#ff7f0e",  # arancione
            "#2ca02c",  # verde
            "#d62728",  # rosso
            "#9467bd",  # viola
            "#8c564b",  # marrone
            "#e377c2"   # rosa
        ]

        cnt_idx = 0
        gen_colors = {}
        for idx in range(len(generation_losses)):
            name = "G" + str(idx+1)
            gen_colors[name] = colors_hex[idx]
            cnt_idx = idx

        n_dist = len(distillation_losses)
        dist_colors = {}
        for idx in range(len(distillation_losses)):
            name = "KD" + str(idx+1)
            dist_colors[name] = colors_hex[(idx+cnt_idx)%n_dist]

        # ------------------------------------------ Pramble: Recap --------------------------------- #
        fig, ax = plt.subplots(figsize=(12,24))
        fig.patch.set_facecolor(colors['bg'])
        ax.axis('off')
        
        # Calcola statistiche finali
        final_G_loss = G_tot_loss_epochs[-1] if G_tot_loss_epochs else 0
        G_spec_losses_final = []
        for spec_loss_epochs in G_spec_losses_epochs:
            if spec_loss_epochs:
                G_spec_losses_final.append(spec_loss_epochs[-1])
            else:
                G_spec_losses_final.append(0)

        final_S_train_loss = S_train_loss_epochs[-1] if S_train_loss_epochs else 0
        final_S_train_acc = S_train_acc_epochs[-1] if S_train_acc_epochs else 0
        final_S_val_loss = S_val_loss_epochs[-1] if S_val_loss_epochs else 0
        final_S_val_acc = S_val_acc_epochs[-1] if S_val_acc_epochs else 0
        
        # Layout ordinato e preciso
        y_start = 0.95
        line_height = 0.020
        section_gap = 0.02
        current_y = y_start
        
        # Titolo principale
        if args.kdci:
            ax.text(0.5, current_y, f'KDCI-{args.method.upper()} TRAINING REPORT', fontsize=24, fontweight='bold', color=colors['generator'], ha='center', va='top')
        else:
            ax.text(0.5, current_y, f'{args.method.upper()} TRAINING REPORT', fontsize=24, fontweight='bold', color=colors['generator'], ha='center', va='top')
        current_y -= 0.04
        
        # Linea separatrice
        ax.plot([0.1, 0.9], [current_y, current_y], color=colors['generator'], linewidth=2)
        current_y -= 0.04
        
        # Sezione Model Information
        ax.text(0.1, current_y, 'MODEL INFORMATION', fontsize=16, fontweight='bold', color=colors['text'])
        current_y -= line_height
        
        if args.ood_loss and args.additive_loss:
            which_ood_loss = "ADDITIVE"
            gamma_ood = args.gamma_ood
        elif args.ood_loss:
            which_ood_loss = "SUBSTITUTE"
            gamma_ood = args.gamma_ood
        else:
            which_ood_loss = "NONE"
            gamma_ood = "-X-"
        
        if args.gamma_adaptive != "none":
            is_gamma_adaptive = args.gamma_adaptive.upper()
            new_gamma = args.new_gamma_ood
        else:
            is_gamma_adaptive = "NO"
            new_gamma = "-X-"

        model_info_data = [
            ('Teacher Model', t_model_name),
            ('Student Model', s_model_name),
            ('Dataset', dataset),
            ('Number of Classes', args.num_classes),
            ('Approach', args.method.upper()),
            ('KDCI', "YES" if args.kdci else "NO"),
            ('OOD loss', which_ood_loss),
            ('Gamma OOD', gamma_ood),
            ('Adaptive Gamma OOD', is_gamma_adaptive),
            ('Adaptive new Gamma OOD', new_gamma),
            ('Generator reset', "YES" if args.g_reset else "NO"),
            ('Generator penalty', "YES" if args.g_penality else "NO"),
            ('Teacher-student energy match', "YES" if args.s_energy_match else "NO")
        ]

        label_x_pos = 0.15
        value_x_pos = 0.55

        for label, value in model_info_data:        
            ax.text(label_x_pos, current_y, f'{label}:', fontsize=11, color=colors['text'], fontweight='bold') 
            value_str = str(value) 
            value_color = colors['text'] 

            if value_str == "YES":
                value_color = 'green'
            elif value_str == "NO" or value_str == "-X-" or value_str == "NONE":
                value_color = 'red'
            elif value_str == "SUBSTITUTE" or value_str == "ADDITIVE":
                value_color = 'blue'
            elif label == 'Adaptive Gamma OOD' and value_str != 'NO':
                value_color = 'blue'
                
            ax.text(value_x_pos, current_y, value_str, fontsize=11, color=value_color)
            current_y -= line_height
        
        current_y -= section_gap
        
        # Sezione Training Configuration
        ax.text(0.1, current_y, 'TRAINING CONFIGURATION', fontsize=16, fontweight='bold', color=colors['text'])
        current_y -= line_height
        
        training_config = [
            f'Total Epochs: {args.epochs}',
            f'Train Batch Size: {args.train_batch_size}',
            f'Test Batch Size: {args.test_batch_size}',
            f'Dataset Size: {args.dataset_size}',
            f'Steps per Epoch: {steps_per_epoch}',
            f'Generator Learning Rate: {args.lr_G}',
            f'Student Learning Rate: {args.lr_S}',
            f'Student Optimizer: SGD (momentum={args.momentum}, weight_decay={args.weight_decay})',
            f'Student Scheduler: CosineAnnealingLR',
            f'Generator Latent Dimension (z_dim): {args.z_dim}'
        ]
        
        for config in training_config:
            ax.text(0.15, current_y, config, fontsize=11, color=colors['text'])
            current_y -= line_height
        
        current_y -= section_gap
        
        # Sezione Training Results
        ax.text(0.1, current_y, 'TRAINING RESULTS', fontsize=16, fontweight='bold', color=colors['text'])
        current_y -= line_height
        
        results = [
            f'Best Student Validation Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)',
            f'Final Generator Total Loss: {final_G_loss:.4f}',
        ]

        for idx in range(len(G_spec_losses)):
            spec_loss_final = G_spec_losses_final[idx]
            spec_loss_name = G_spec_losses[idx]["name"]
            color_name = "L" + str(idx+1)
            label_name = color_name + " Loss (" + spec_loss_name + ")"

            results += [f'Generator {label_name}: {spec_loss_final:.4f}']
        
        OOD_loss_final = generation_losses[0]["values"][-1]
        OOD_loss_name = generation_losses[0]["name"]
        color_name = "G1"
        label_name = color_name + " Loss (" + OOD_loss_name + ")"

        results += [f'Generator {label_name}: {OOD_loss_final:.4f}']

        results += [
            f'Final Student Training Loss (KD): {final_S_train_loss:.4f}',
            f'Final Student Training Accuracy: {final_S_train_acc:.4f} ({final_S_train_acc*100:.2f}%)' if final_S_train_acc > 0 else 'Final Student Training Accuracy: N/A',
            f'Final Student Validation Loss: {final_S_val_loss:.4f}' if final_S_val_loss > 0 else 'Final Student Validation Loss: N/A',
            f'Final Student Validation Accuracy: {final_S_val_acc:.4f} ({final_S_val_acc*100:.2f}%)' if final_S_val_acc > 0 else 'Final Student Validation Accuracy: N/A'
        ]
        
        for result in results:
            ax.text(0.15, current_y, result, fontsize=11, color=colors['text'])
            current_y -= line_height
        
        current_y -= section_gap
        
        # Sezione Statistics
        ax.text(0.1, current_y, 'STATISTICS', fontsize=16, fontweight='bold', color=colors['text'])
        current_y -= line_height
        
        val_batches_per_epoch_s_val = len(history["S_val_loss"]) // args.epochs if "S_val_loss" in history and len(history["S_val_loss"]) > 0 else 0
        
        statistics = [
            f'Total Generator Training Steps: {len(G_train_loss)}',
            f'Total Student Training Steps: {len(S_train_loss)}',
            f'Total Student Training Accuracy Points: {len(history["S_train_accuracy"]) if "S_train_accuracy" in history else 0}',
            f'Total Student Validation Loss Points: {len(history["S_val_loss"]) if "S_val_loss" in history else 0}',
            f'Total Student Validation Accuracy Points: {len(history["S_val_accuracy"]) if "S_val_accuracy" in history else 0}',
            f'Validation Batches per Epoch: {val_batches_per_epoch_s_val}',
            "\n",
        ]

        for stat in statistics:
            ax.text(0.15, current_y, stat, fontsize=11, color=colors['text'])
            current_y -= line_height

        summary = [
            f'  • Training Convergence: {"Good" if final_S_train_loss < 1.0 else "Needs Improvement"}',
            f'  • Validation Performance: {"Good" if final_S_val_acc > 0.7 else "Needs Improvement" if final_S_val_acc > 0 else "N/A"}',
            f'  • Overfitting Check: {"Possible Overfitting" if final_S_train_acc > 0 and final_S_val_acc > 0 and (final_S_train_acc - final_S_val_acc) > 0.1 else "Good Generalization" if final_S_train_acc > 0 and final_S_val_acc > 0 else "N/A"}',
            f'  • Generator Stability: {"Stable" if len(G_tot_loss_epochs) > 5 and abs(G_tot_loss_epochs[-1] - G_tot_loss_epochs[-5]) < 0.5 else "Unstable" if len(G_tot_loss_epochs) > 5 else "N/A"}',
            "\n\n"
        ]
        
        ax.text(0.15, current_y, 'Performance Summary:', fontsize=12, color=colors['text'], weight="bold")
        current_y -= line_height
        for s in summary:
            ax.text(0.15, current_y, s, fontsize=11, color=colors['text'])
            current_y -= line_height

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        pdf.savefig(fig, bbox_inches='tight', facecolor=colors['bg'])
        plt.close()
        
        # ------------------------------------------ Pagina 1: Generation Method Losses --------------------------------- #
        if len(G_spec_losses) == 3:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 12))
            axes = [ax1, ax2, ax3, ax4]
        elif len(G_spec_losses) == 2:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
            axes = [ax1, ax2, ax3]
        else:
            fig, ((ax1, ax2, ax3), (ax_empty, ax4, ax5)) = plt.subplots(2, 3, figsize=(24, 12))
            ax_empty.axis('off')  # Nasconde l'ultimo subplot vuoto
            axes = [ax1, ax2, ax3, ax4, ax5]
        
        # Converti step in epoche per i grafici step-wise
        G_step_epochs = np.linspace(1, args.epochs, len(G_train_loss))
        fig.patch.set_facecolor(colors['bg'])

        # Generator Total Loss
        ax1.plot(G_step_epochs, G_train_loss, color=colors['generator'], alpha=0.3, linewidth=1, label='Per Step')
        ax1.plot(epochs_range, G_tot_loss_epochs, color=colors['generator'], linewidth=3, marker='o', markersize=4, alpha=0.9, label='Per Epoch')
        ax1.set_title('Generator Total Loss', fontsize=16, fontweight='bold', color=colors['text'], pad=15)
        ax1.set_xlabel('Epochs', fontsize=12, color=colors['text'])
        ax1.set_ylabel('Loss', fontsize=12, color=colors['text'])
        ax1.grid(True, alpha=0.3, color=colors['grid'])
        ax1.set_facecolor(colors['bg'])
        ax1.tick_params(colors=colors['text'])
        ax1.legend(frameon=True, fontsize=10)
        
        # Generator Specific Losses
        for idx in range(len(G_spec_losses)):
            spec_loss_values = G_spec_losses[idx]["values"]
            spec_loss_name   = G_spec_losses[idx]["name"]
            color_name = "L" + str(idx+1)
            color = colors[color_name]
            spec_loss_epochs = G_spec_losses_epochs[idx]

            axes[idx+1].plot(G_step_epochs, spec_loss_values, color=color, alpha=0.3, linewidth=1, label='Per Step')
            axes[idx+1].plot(epochs_range, spec_loss_epochs, color=color, linewidth=3, marker='s', markersize=4, alpha=0.9, label='Per Epoch')
            axes[idx+1].set_title(f'Generator Loss {spec_loss_name}', fontsize=16, fontweight='bold', color=colors['text'], pad=15)
            axes[idx+1].set_xlabel('Epochs', fontsize=12, color=colors['text'])
            axes[idx+1].set_ylabel('Loss', fontsize=12, color=colors['text'])
            axes[idx+1].grid(True, alpha=0.3, color=colors['grid'])
            axes[idx+1].set_facecolor(colors['bg'])
            axes[idx+1].tick_params(colors=colors['text'])
            axes[idx+1].legend(frameon=True, fontsize=10)
        
        plt.suptitle(f'{args.method.upper()} Generator Losses - {t_model_name} → {s_model_name}', fontsize=20, fontweight='bold', color=colors['text'], y=0.99)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', facecolor=colors['bg'])
        plt.close()
        
        # ------------------------------------------ Pagina 2: Generation Stats --------------------------------- #
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 12))
        axes = [ax1, ax2, ax3, ax4]
            
        for idx in range(len(generation_losses)):
            generation_loss_values = generation_losses[idx]["values"]
            generation_loss_name   = generation_losses[idx]["name"]

            print(str(generation_loss_name) + ": " + str(len(generation_loss_values)))
            
        for idx in range(len(generation_losses)):
            generation_loss_values = generation_losses[idx]["values"]
            generation_loss_name   = generation_losses[idx]["name"]
            color_name = "G" + str(idx+1)
            color = gen_colors[color_name]
            generation_loss_epochs = generation_losses_epochs[idx]

            axes[idx].plot(G_step_epochs, generation_loss_values, color=color, alpha=0.3, linewidth=1, label='Per Step')
            axes[idx].plot(epochs_range, generation_loss_epochs, color=color, linewidth=3, marker='s', markersize=4, alpha=0.9, label='Per Epoch')
            axes[idx].set_title(generation_loss_name, fontsize=16, fontweight='bold', color=colors['text'], pad=15)
            axes[idx].set_xlabel('Epochs', fontsize=12, color=colors['text'])
            axes[idx].set_ylabel('Value', fontsize=12, color=colors['text'])
            axes[idx].grid(True, alpha=0.3, color=colors['grid'])
            axes[idx].set_facecolor(colors['bg'])
            axes[idx].tick_params(colors=colors['text'])
            axes[idx].legend(frameon=True, fontsize=10)

        plt.suptitle(f'{args.method.upper()} Generation Process Analysis - {t_model_name} → {s_model_name}', fontsize=20, fontweight='bold', color=colors['text'], y=0.99)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', facecolor=colors['bg'])
        plt.close()

        # ------------------------------------------ Pagina 4: Generated samples --------------------------------- #
        fig = plt.figure(figsize=(24, 12))
        fig.patch.set_facecolor(colors['bg'])

        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], hspace=0.2, wspace=0.25, left=0.05, right=0.98)

        # Synthetic samples distribution
        hist_ax = fig.add_subplot(gs[0, 0])
        hist_ax.set_title(f'Pseudo-label Distribution (over {args.dataset_size // 5} samples)', 
                        fontsize=16, fontweight='bold', color=colors['text'], pad=15)
        labels = [f"{i}" for i in range(args.num_classes)]
        pseudo_label_counts = [label_count[i] if i in label_count else 0 for i in range(args.num_classes)]

        hist_ax.bar(labels, pseudo_label_counts, color='skyblue')
        hist_ax.set_xlabel('Pseudo-label', fontsize=12, color=colors['text'])
        hist_ax.set_ylabel('Count', fontsize=12, color=colors['text'])

        if args.num_classes > 10:
            step = 10
            major_ticks = np.arange(0, args.num_classes + 1, step)
            hist_ax.set_xticks(major_ticks)
            hist_ax.set_xticklabels(major_ticks, rotation=0, fontsize=10)
        else:
            x_pos = np.arange(args.num_classes)
            hist_ax.set_xticks(x_pos)
            hist_ax.set_xticklabels(x_pos, rotation=0, fontsize=10)

        hist_ax.set_xlim(-1, args.num_classes)
        hist_ax.tick_params(axis='x', rotation=45, labelsize=10, colors=colors['text'])
        hist_ax.tick_params(axis='y', labelsize=10, colors=colors['text'])
        hist_ax.grid(True, alpha=0.3, color=colors['grid'])
        hist_ax.set_facecolor(colors['bg'])

        # Synthetic samples examples
        samples_ax = fig.add_subplot(gs[:, 1])
        samples_ax.axis('off')
        samples_ax.set_title('Synthetic Samples Examples', 
                            fontsize=14, fontweight='bold', color=colors['text'], pad=10)

        if sample_images and len(sample_images) > 0:
            num_images = min(len(sample_images), 12)
            rows, cols = 3, 4
            
            inner_gs = GridSpecFromSubplotSpec(rows, cols, subplot_spec=gs[:, 1], hspace=0.3, wspace=0.3)
            
            for i in range(rows * cols):
                ax = fig.add_subplot(inner_gs[i])
                ax.set_facecolor(colors['bg'])
                
                if i < num_images:
                    try:
                        img_path, pseudo_label = sample_images[i]
                        img = plt.imread(img_path)
                        ax.imshow(img)
                        ax.set_title(f'Pseudo-label: {pseudo_label}', 
                                fontsize=10, fontweight='bold', color=colors['text'])
                        ax.axis('off')
                    except Exception as e:
                        ax.text(0.5, 0.5, f'Image {i+1}\nError loading', 
                            ha='center', va='center', fontsize=10, color=colors['text'])
                        ax.set_xlim(0, 1)
                        ax.set_ylim(0, 1)
                        ax.axis('off')
                else:
                    ax.axis('off')

        plt.suptitle(f'Sample Generated Images - {t_model_name} → {s_model_name}', 
                    fontsize=20, fontweight='bold', color=colors['text'], y=0.96)

        pdf.savefig(fig, facecolor=colors['bg'], dpi=100)
        plt.close(fig)

        # ------------------------------------------ Pagina 3: Student metrics --------------------------------- #
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 12))
        fig.patch.set_facecolor(colors['bg'])
        
        # Student Training Loss
        S_train_step_epochs = np.linspace(1, args.epochs, len(S_train_loss))
        ax1.plot(S_train_step_epochs, S_train_loss, color=colors['train'], alpha=0.3, linewidth=1, label='Per Step')
        ax1.plot(epochs_range, S_train_loss_epochs, color=colors['train'], linewidth=3, marker='o', markersize=4, alpha=0.9, label='Per Epoch')
        ax1.set_title('Student Training Loss (KD)', fontsize=16, fontweight='bold', color=colors['text'], pad=15)
        ax1.set_xlabel('Epochs', fontsize=12, color=colors['text'])
        ax1.set_ylabel('Loss', fontsize=12, color=colors['text'])
        ax1.grid(True, alpha=0.3, color=colors['grid'])
        ax1.set_facecolor(colors['bg'])
        ax1.tick_params(colors=colors['text'])
        ax1.legend(frameon=True, fontsize=10)
        
        # Student Training Accuracy
        if S_train_acc_epochs:
            S_train_acc_step_epochs = np.linspace(1, args.epochs, len(S_train_accuracy))
            ax2.plot(S_train_acc_step_epochs, S_train_accuracy, color=colors['train'], alpha=0.3, linewidth=1, label='Per Step')
            ax2.plot(epochs_range, S_train_acc_epochs, color=colors['train'], linewidth=3, marker='s', markersize=4, alpha=0.9, label='Per Epoch')
            ax2.legend(frameon=True, fontsize=10)
        ax2.set_title('Student Training Accuracy', fontsize=16, fontweight='bold', color=colors['text'], pad=15)
        ax2.set_xlabel('Epochs', fontsize=12, color=colors['text'])
        ax2.set_ylabel('Accuracy', fontsize=12, color=colors['text'])
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3, color=colors['grid'])
        ax2.set_facecolor(colors['bg'])
        ax2.tick_params(colors=colors['text'])
        
        # Student Validation Loss
        if S_val_loss_epochs:
            S_val_step_epochs = np.linspace(1, args.epochs, len(S_val_loss))
            ax3.plot(S_val_step_epochs, S_val_loss, color=colors['val'], alpha=0.3, linewidth=1, label='Per Batch')
            ax3.plot(epochs_range, S_val_loss_epochs, color=colors['val'], linewidth=3, marker='^', markersize=4, alpha=0.9, label='Per Epoch')
            ax3.legend(frameon=True, fontsize=10)
        ax3.set_title('Student Validation Loss', fontsize=16, fontweight='bold', color=colors['text'], pad=15)
        ax3.set_xlabel('Epochs', fontsize=12, color=colors['text'])
        ax3.set_ylabel('Loss', fontsize=12, color=colors['text'])
        ax3.grid(True, alpha=0.3, color=colors['grid'])
        ax3.set_facecolor(colors['bg'])
        ax3.tick_params(colors=colors['text'])
        
        # Student Validation Accuracy
        if S_val_acc_epochs:
            S_val_acc_step_epochs = np.linspace(1, args.epochs, len(history["S_val_accuracy"]))
            ax4.plot(S_val_acc_step_epochs, history["S_val_accuracy"], color=colors['val'], alpha=0.3, linewidth=1, label='Per Batch')
            ax4.plot(epochs_range, S_val_acc_epochs, color=colors['val'], linewidth=3, marker='d', markersize=4, alpha=0.9, label='Per Epoch')
            ax4.legend(frameon=True, fontsize=10)
        ax4.set_title('Student Validation Accuracy', fontsize=16, fontweight='bold', color=colors['text'], pad=15)
        ax4.set_xlabel('Epochs', fontsize=12, color=colors['text'])
        ax4.set_ylabel('Accuracy', fontsize=12, color=colors['text'])
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3, color=colors['grid'])
        ax4.set_facecolor(colors['bg'])
        ax4.tick_params(colors=colors['text'])
        
        plt.suptitle(f'{args.method.upper()} Student Complete Metrics - {t_model_name} → {s_model_name}', fontsize=20, fontweight='bold', color=colors['text'], y=0.99)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', facecolor=colors['bg'])
        plt.close()
        
        # ------------------------------------------ Pagina 4: Distillation vs Generation --------------------------------- #
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 12))
        fig.patch.set_facecolor(colors['bg'])
        
        # Loss Comparison: Training vs Validation
        ax1.plot(epochs_range, S_train_loss_epochs, color=colors['train'], linewidth=3, marker='o', markersize=4, label='Training Loss', alpha=0.9)
        if S_val_loss_epochs:
            ax1.plot(epochs_range, S_val_loss_epochs, color=colors['val'], linewidth=3, marker='s', markersize=4, label='Validation Loss', alpha=0.9)
        ax1.set_title('Student Loss: Training vs Validation', fontsize=16, fontweight='bold', color=colors['text'], pad=15)
        ax1.set_xlabel('Epochs', fontsize=12, color=colors['text'])
        ax1.set_ylabel('Loss', fontsize=12, color=colors['text'])
        ax1.grid(True, alpha=0.3, color=colors['grid'])
        ax1.set_facecolor(colors['bg'])
        ax1.tick_params(colors=colors['text'])
        ax1.legend(frameon=True, fontsize=10)
        
        # Accuracy Comparison: Training vs Validation
        if S_train_acc_epochs:
            ax2.plot(epochs_range, S_train_acc_epochs, color=colors['train'], linewidth=3, marker='o', markersize=4, label='Training Accuracy', alpha=0.9)
        if S_val_acc_epochs:
            ax2.plot(epochs_range, S_val_acc_epochs, color=colors['val'], linewidth=3, marker='s', markersize=4, label='Validation Accuracy', alpha=0.9)
        ax2.set_title('Student Accuracy: Training vs Validation', fontsize=16, fontweight='bold', color=colors['text'], pad=15)
        ax2.set_xlabel('Epochs', fontsize=12, color=colors['text'])
        ax2.set_ylabel('Accuracy', fontsize=12, color=colors['text'])
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3, color=colors['grid'])
        ax2.set_facecolor(colors['bg'])
        ax2.tick_params(colors=colors['text'])
        ax2.legend(frameon=True, fontsize=10)
        
        # Loss Components Comparison (Generator)
        markers = ['o','s','^','+']
        for idx in range(len(G_spec_losses)):
            spec_loss_epochs = G_spec_losses_epochs[idx]
            spec_loss_name   = G_spec_losses[idx]["name"]
            color_name = "L" + str(idx+1)
            label_name = color_name + " (" + spec_loss_name + ")"
            color = colors[color_name]
            marker = markers[idx]

            ax3.plot(epochs_range, spec_loss_epochs, color=color, linewidth=2, marker=marker, markersize=3, label=label_name, alpha=0.8)

        ax3.set_title('Generator Loss Components', fontsize=16, fontweight='bold', color=colors['text'], pad=15)
        ax3.set_xlabel('Epochs', fontsize=12, color=colors['text'])
        ax3.set_ylabel('Loss', fontsize=12, color=colors['text'])
        ax3.grid(True, alpha=0.3, color=colors['grid'])
        ax3.set_facecolor(colors['bg'])
        ax3.tick_params(colors=colors['text'])
        ax3.legend(frameon=True, fontsize=10)
        
        # Overall Performance Summary
        ax4.plot(epochs_range, G_tot_loss_epochs, color=colors['generator'], linewidth=2, marker='o', markersize=3, label='Generator Total Loss', alpha=0.8)
        ax4.plot(epochs_range, S_train_loss_epochs, color=colors['train'], linewidth=2, marker='s', markersize=3, label='Student Train Loss', alpha=0.8)
        if S_val_loss_epochs:
            ax4.plot(epochs_range, S_val_loss_epochs, color=colors['val_comp'], linewidth=2, marker='^', markersize=3, label='Student Val Loss', alpha=0.8)
        ax4.set_title('Overall Training Summary', fontsize=16, fontweight='bold', color=colors['text'], pad=15)
        ax4.set_xlabel('Epochs', fontsize=12, color=colors['text'])
        ax4.set_ylabel('Loss', fontsize=12, color=colors['text'])
        ax4.grid(True, alpha=0.3, color=colors['grid'])
        ax4.set_facecolor(colors['bg'])
        ax4.tick_params(colors=colors['text'])
        ax4.legend(frameon=True, fontsize=10)
        
        plt.suptitle(f'{args.method.upper()} Training Analysis - {t_model_name} → {s_model_name}', fontsize=20, fontweight='bold', color=colors['text'], y=0.99)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', facecolor=colors['bg'])
        plt.close()
        
        # ------------------------------------------ Pagina 5: Distillation metrics --------------------------------- #
        fig, ((ax1, ax2, ax3), (ax_empty, ax4, ax5)) = plt.subplots(2, 3, figsize=(24, 12))
        ax_empty.axis('off')  # Nasconde l'ultimo subplot vuoto
        axes = [ax1, ax2, ax3, ax4, ax5]

        for idx in range(len(distillation_losses)):
            distillation_loss_values = distillation_losses[idx]["values"]
            distillation_loss_name   = distillation_losses[idx]["name"]
            color_name = "KD" + str(idx+1)
            color = dist_colors[color_name]
            distillation_loss_epochs = distillation_losses_epochs[idx]

            axes[idx].plot(S_train_step_epochs, distillation_loss_values, color=color, alpha=0.3, linewidth=1, label='Per Step')
            axes[idx].plot(epochs_range, distillation_loss_epochs, color=color, linewidth=3, marker='s', markersize=4, alpha=0.9, label='Per Epoch')
            axes[idx].set_title(distillation_loss_name, fontsize=16, fontweight='bold', color=colors['text'], pad=15)
            axes[idx].set_xlabel('Epochs', fontsize=12, color=colors['text'])
            axes[idx].set_ylabel('Value', fontsize=12, color=colors['text'])
            axes[idx].grid(True, alpha=0.3, color=colors['grid'])
            axes[idx].set_facecolor(colors['bg'])
            axes[idx].tick_params(colors=colors['text'])
            axes[idx].legend(frameon=True, fontsize=10)

        plt.suptitle(f'{args.method.upper()} Distillation Process Analysis - {t_model_name} → {s_model_name}', fontsize=20, fontweight='bold', color=colors['text'], y=0.99)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', facecolor=colors['bg'])
        plt.close()
    
    print(f"[LOG] {args.method.upper()} Training report PDF saved: {pdf_filename}")

def extract_features(dataloader, device, model, data_name, extr_feat, max_samples=1000):
    model.eval()
    features, energies = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting energy from {data_name} samples"):
            imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
            imgs = imgs.to(device)
            if extr_feat:
                feats, _ = model(imgs)
            else:
                feats = model(imgs)
            energy = -torch.logsumexp(feats, dim=1)
            features.append(feats.cpu())
            energies.append(energy.cpu())

    features = torch.cat(features, dim=0)
    energies = torch.cat(energies, dim=0)

    # Sampling
    if len(features) > max_samples:
        idx = torch.randperm(len(features))[:max_samples]
        features = features[idx]
        energies = energies[idx]

    return features.numpy(), energies.numpy()

def calculate_coverage_score(X_real, X_synth, radius_factor=0.05):
    """
    Calcola il Coverage Score per t-SNE.
    
    Args:
        X_real: coordinate t-SNE dei dati reali
        X_synth: coordinate t-SNE dei dati sintetici  
        radius_factor: fattore per definire il raggio (percentuale della std)
    
    Returns:
        float: Coverage score (0-1, più alto è meglio)
    """
    # Calcola il raggio come percentuale della deviazione standard
    std_real = np.std(X_real, axis=0)
    radius = radius_factor * np.mean(std_real)
    
    # Per ogni punto reale, controlla se ha almeno un sintetico entro il raggio
    distances = cdist(X_real, X_synth)
    covered = np.any(distances <= radius, axis=1)
    
    coverage = np.mean(covered)
    return coverage

def calculate_wasserstein_distance(X_real, X_synth):
    """
    Calcola la Wasserstein Distance approssimata per UMAP.
    
    Args:
        X_real: coordinate UMAP dei dati reali
        X_synth: coordinate UMAP dei dati sintetici
    
    Returns:
        float: Wasserstein distance (più basso è meglio)
    """
    # Approssimazione usando il transport cost tra le medie delle distribuzioni
    # Per una versione più accurata potresti usare scipy.stats.wasserstein_distance per 1D
    # o POT (Python Optimal Transport) per 2D
    
    from scipy.stats import wasserstein_distance
    
    # Calcola Wasserstein per ogni dimensione e poi fa la media
    w_dist_x = wasserstein_distance(X_real[:, 0], X_synth[:, 0])
    w_dist_y = wasserstein_distance(X_real[:, 1], X_synth[:, 1])
    
    return (w_dist_x + w_dist_y) / 2

def calculate_jensen_shannon_divergence(energy_real, energy_synth, bins=50):
    """
    Calcola la Jensen-Shannon Divergence per le distribuzioni energetiche.
    
    Args:
        energy_real: valori energetici dei dati reali
        energy_synth: valori energetici dei dati sintetici
        bins: numero di bin per l'istogramma
    
    Returns:
        float: JS divergence (0-1, più basso è meglio)
    """
    # Crea istogrammi normalizzati
    min_val = min(energy_real.min(), energy_synth.min())
    max_val = max(energy_real.max(), energy_synth.max())
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    
    hist_real, _ = np.histogram(energy_real, bins=bin_edges, density=True)
    hist_synth, _ = np.histogram(energy_synth, bins=bin_edges, density=True)
    
    # Normalizza per ottenere distribuzioni di probabilità
    hist_real = hist_real / hist_real.sum()
    hist_synth = hist_synth / hist_synth.sum()
    
    # Evita log(0) aggiungendo un piccolo epsilon
    epsilon = 1e-10
    hist_real = hist_real + epsilon
    hist_synth = hist_synth + epsilon
    
    # Calcola JS divergence
    m = (hist_real + hist_synth) / 2
    js_div = (entropy(hist_real, m) + entropy(hist_synth, m)) / 2
    
    return js_div

def plot_tsne(X1, X2, path, dataset_name, syn_name):  
    """
    Coverage Score - Calcola la percentuale di campioni reali (blu) che hanno almeno un campione sintetico (arancione) 
    entro un raggio definito (es. 5% della deviazione standard della distribuzione). 
    Questa metrica è perfetta per t-SNE perché valuta quanto bene i dati sintetici 
    "coprono" lo spazio dei dati reali, che è esattamente quello che vuoi vedere visivamente.

    THE LOWER THE BETTER
    """
    X = np.concatenate([X1, X2])
    y = np.array([0]*len(X1) + [1]*len(X2))
    X_emb = TSNE(n_components=2, perplexity=30).fit_transform(X)

    # Calcola Coverage Score sulle coordinate t-SNE
    coverage = calculate_coverage_score(X_emb[y==0], X_emb[y==1])

    plt.figure()
    plt.scatter(X_emb[y==0,0], X_emb[y==0,1], label=dataset_name, alpha=0.6)
    plt.scatter(X_emb[y==1,0], X_emb[y==1,1], label=syn_name, alpha=0.6)
    plt.legend()
    plt.title(f"t-SNE - Coverage (↓): {coverage:.3f}")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_umap(X1, X2, path, dataset_name, syn_name):
    X = np.concatenate([X1, X2])
    y = np.array([0]*len(X1) + [1]*len(X2))
    X_emb = umap.UMAP(random_state=42).fit_transform(X)

    # Calcola Wasserstein Distance sulle coordinate UMAP
    w_distance = calculate_wasserstein_distance(X_emb[y==0], X_emb[y==1])

    plt.figure()
    plt.scatter(X_emb[y==0,0], X_emb[y==0,1], label=dataset_name, alpha=0.6)
    plt.scatter(X_emb[y==1,0], X_emb[y==1,1], label=syn_name, alpha=0.6)
    plt.legend()
    plt.title(f"UMAP - Wasserstein (↓): {w_distance:.3f}")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_energy(e1, e2, path, dataset_name, syn_name, comp_targ=False, real_energy_mean=0.0, estimated_energy_means={}):
    # Calcola Jensen-Shannon Divergence
    if not comp_targ:
        js_div = calculate_jensen_shannon_divergence(e1, e2)

        plt.figure()
        sns.histplot(e1, label=dataset_name, kde=True, bins=50)
        sns.histplot(e2, label=syn_name, kde=True, bins=50)
        plt.legend()
        plt.title(f"Energy Distribution - JS Div (↓): {js_div:.3f}")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
    else:
        # plt.figure()
        # sns.histplot(e1, label=dataset_name, kde=True, bins=50)
        # plt.legend()
        # plt.title(f"{dataset_name.upper()} Energy Distribution - Real {real_energy_mean:.2f}")
        # plt.tight_layout()
        # plt.savefig(path)
        # plt.close()

        plt.figure(figsize=(10, 6))

        sns.histplot(e1, label=dataset_name, kde=True, bins=50, color='lightgray', alpha=0.5)
        plt.axvline(x=real_energy_mean, color='red', linestyle='--', linewidth=2.5, label=f'True Mean ({real_energy_mean:.2f})')

        colors = sns.color_palette("tab10", len(estimated_energy_means))

        for (method, val), color in zip(estimated_energy_means.items(), colors):
            plt.axvline(
                x=val, 
                color=color, 
                linestyle='-',
                linewidth=2, 
                label=f'{method} ({val:.2f})'
            )

        subtitle_items = [f"{m}: {v:.2f}" for m, v in estimated_energy_means.items()]
        values_str = ", ".join(subtitle_items)

        main_title = f"{dataset_name.upper()} Energy Distribution - Real energy mean: {real_energy_mean:.2f}"
        plt.title(f"{main_title}\nEstimates: [{values_str}]", fontsize=11) 

        plt.legend(loc='best') 
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

def save_image_to_pdf(image_path, pdf):
    img = plt.imread(image_path)
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    pdf.savefig()
    plt.close()

def save_result_json(args, dataset, method, method_str, teacher_name, student_name, best_acc, total_training_time, training_time_str, memory_usage, results_dir="work/project/results/",test=True):
    """
    Salva risultato in file separato (thread-safe).
    """
    # Crea directory
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    teacher_name = teacher_name.lower()
    student_name = student_name.lower()

    # Nome file unico con timestamp per evitare conflitti
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # millisecondi
    container_id = os.environ.get('HOSTNAME', 'unknown')[:8]  # primi 8 char
    if test:
        filename = f"test{method}_{dataset}_{teacher_name}_{student_name}{method_str}_{timestamp}_{container_id}.json" 
    else:
        filename = f"{method}_{dataset}_{teacher_name}_{student_name}{method_str}_{timestamp}_{container_id}.json" 

    result = {
        "dataset": dataset,
        "method": method,
        "method_str": method_str,
        "teacher": teacher_name.lower(),
        "student": student_name,
        "accuracy": float(best_acc),
        "lr_G": float(args.lr_G),
        "train_time_sec": round(float(total_training_time), 2),
        "train_time": training_time_str,
        "memory_usage": memory_usage
    }

    if method.lower() == "dafl":
        result.update({
            "oh": float(args.oh),
            "act": float(args.act),
            "ie": float(args.ie)
        })
    elif method.lower() == "fast":
        result.update({
            "oh": float(args.oh),
            "adv": float(args.adv),
            "feat": float(args.feat)
        })

    # Salva in file separato (zero conflitti)
    filepath = Path(results_dir) / filename
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)
    
    # Log come prima
    out_txt = f"{method}| {dataset} | {method_str} | {teacher_name.lower()} → {student_name} | {best_acc:.3f}"
    print(f"[RESULT] {out_txt}")
    print(f"[SAVED] {filepath}")

def get_gpu_memory_usage():
    if not torch.cuda.is_available():
        return 0, 0
    
    # Sincronizza per assicurarsi che tutte le operazioni siano finite
    torch.cuda.synchronize()
    
    # 1. Max Allocated: Il picco massimo di memoria usata dai tensori
    # (È la misura più precisa per capire se il modello "ci sta" nella GPU)
    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    # 2. Reserved: La memoria totale che PyTorch ha chiesto al sistema operativo
    # (Questo è il valore che vedi di solito su nvidia-smi)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    
    return max_allocated, reserved

def estimate_energy_from_bn_stats(teacher, num_classes=10):
    """
    Usa BN stats come indicatore qualitativo, non quantitativo.
    """
    
    # Estrai BN stats
    bn_scales = []
    for module in teacher.modules():
        if isinstance(module, nn.BatchNorm2d):
            # La "scala" effettiva delle attivazioni
            scale = (module.running_var.sqrt() * module.weight).abs().mean().item()
            bn_scales.append(scale)
    
    if len(bn_scales) == 0:
        return -9.5  # Fallback
    
    # Media delle scale
    avg_scale = np.mean(bn_scales)
    
    # Formula empirica calibrata per ResNet:
    # avg_scale tipicamente in [0.5, 2.0]
    # Mappiamo a energia in [-11, -8]
    
    # Regressione lineare inversa:
    # scale=0.5 → energia=-11
    # scale=2.0 → energia=-8
    
    energy = -11 + (avg_scale - 0.5) * (3 / 1.5)
    
    # Clipping di sicurezza
    energy = np.clip(energy, -12, -7)
    
    print(f"[BN V2] Avg BN scale: {avg_scale:.4f} → Energy: {energy:.2f}")
    return energy + 2.0

def estimate_energy_theoretical(teacher, num_classes=10, num_samples=1000):
    # """
    # Stima teorica CORRETTA: simula softmax output realistico.
    
    # Invece di assumere confidence=0.9, calcoliamo l'energia che
    # corrisponde a un softmax tipico per un teacher ben trainato.
    # """
    
    # # Step 1: Genera logits "tipici" per un classifier ben trainato
    # # Un ResNet su CIFAR-10 tipicamente ha:
    # # - Logit max class: ~10-15
    # # - Altri logits: ~0-5
    
    # device = next(teacher.parameters()).device
    # energies = []
    
    # with torch.no_grad():
    #     for _ in range(num_samples // 100):
    #         # Simula logits realistici
    #         batch_size = 100
    #         logits = torch.zeros(batch_size, num_classes, device=device)
            
    #         # Per ogni sample, scegli una classe random come "corretta"
    #         for i in range(batch_size):
    #             correct_class = np.random.randint(0, num_classes)
                
    #             # Logit della classe corretta: alto (10-15)
    #             logits[i, correct_class] = np.random.uniform(10, 15)
                
    #             # Altri logits: bassi (0-5)
    #             for j in range(num_classes):
    #                 if j != correct_class:
    #                     logits[i, j] = np.random.uniform(0, 5)
            
    #         # Calcola energia
    #         energy = -torch.logsumexp(logits, dim=1)
    #         energies.append(energy)
    
    # energies = torch.cat(energies)
    # estimated_energy = energies.mean().item()
    
    # print(f"[THEORETICAL V2] Estimated energy: {estimated_energy:.2f}")
    # return estimated_energy + 2.0
    energy_target = -9.5 - 0.5 * np.log10(num_classes)
    return energy_target