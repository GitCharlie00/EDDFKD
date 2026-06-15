import glob
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    confusion_matrix, classification_report, precision_recall_fscore_support,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import seaborn as sns
import warnings
import time
import os
from itertools import cycle

from tqdm import tqdm
from datasets_local.dataloaders import init_train_test_loader
from networks.t_networks import get_teacher_model
from networks.s_networks import get_student_model
from utils import *

warnings.filterwarnings("ignore", category=UserWarning, module='torchvision')
warnings.filterwarnings("ignore", category=UserWarning, module='torch.nn')

def test(args,user_suffix):
    ### ---------------------------------- Dataset loading ---------------------------------- ###
    torch.cuda.empty_cache()
    formatter = TextFormatter()
    data_str = formatter.format(
        f"Initializing dataset {args.dataset} for testing",
        color="blue", 
        style=["bold"], 
        separator=True
    )
    print(data_str)

    # Initialize data loaders
    train_loader, test_loader, train_dataset, test_dataset = init_train_test_loader(
        dataset_type=args.dataset,
        dataset_root=args.dataset_root,
        train_batch=args.train_batch_size,
        test_batch=args.test_batch_size,
        num_workers=args.num_workers
    )
    
    print(f"[LOG] Dataset type: {args.dataset}")
    print(f"[LOG] Test dataset size: {len(test_dataset)}")
        
    ### -------------------------------- Model initialization -------------------------------- ###
    model_str = formatter.format(
        "Initializing model for testing",
        color="blue", 
        style=["bold"], 
        separator=True
    )
    print(model_str)

    if "_" in args.network:
        approach_suffix = approach_label(args) + user_suffix
        model, model_name = get_student_model(args.network, args.num_classes)
        _, teacher_name   = get_teacher_model(args.t_network, args.num_classes, args.feature_extr)
    else:
        model, model_name = get_teacher_model(args.network, args.num_classes)
    
    model.to(args.gpu)
    print(f"[LOG] Model: {model_name}")

    ### -------------------------------- Load trained model -------------------------------- ###
    # Single checkpoint file
    if "_" in args.network:
        pattern = f"{args.save_path}{args.dataset}/{teacher_name.lower()}_{model_name.lower()}_{args.dataset.lower()}_{args.approach.lower()}_{args.method.lower()}{approach_suffix}_best_*.pth"
    else:
        pattern = f"{args.save_path}{args.dataset}/{model_name.lower()}_{args.dataset.lower()}_best_*.pth"

    matches = glob.glob(pattern)

    if len(matches) == 0:
        raise FileNotFoundError(f"No checkpoint found at: {pattern}")
            
    checkpoint_path = matches[0]
    
    load_str = formatter.format(
        f"Loading trained model",
        color="blue", 
        style=["bold"], 
        separator=True
    )
    print(load_str)
    print(f"[LOG] Model file: {checkpoint_path}")
    
    # Load the trained weights
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=args.gpu)["state_dict"])
    except:
        model.load_state_dict(torch.load(checkpoint_path, map_location=args.gpu))
    model.eval()

    ### -------------------------------- Testing initialization -------------------------------- ###
    test_str = formatter.format(
        f"Starting testing of {model_name}",
        color="blue", 
        style=["bold"], 
        separator=True
    )
    print(test_str)

    criterion = nn.CrossEntropyLoss()
    
    # Initialize metrics storage
    all_predictions = []
    all_labels = []
    all_probabilities = []
    test_losses = []
    
    # Get class names from the dataset
    class_names = test_dataset.classes

    ### ------------------------------------- Test loop ------------------------------------- ###
    start_time = time.time()
    
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="Testing", leave=True)
        for batch_idx, (images, labels, _) in enumerate(test_bar):
            images = images.to(args.gpu)
            labels = labels.to(args.gpu)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Get probabilities and predictions
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            # Store results
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            test_losses.append(loss.item())
            
            # Calculate running accuracy
            correct = (predicted == labels).sum().item()
            accuracy = correct / labels.size(0)
            
            # Update progress bar
            test_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{accuracy:.4f}")

    torch.cuda.synchronize()
    test_time = time.time() - start_time
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    ### -------------------------------- Calculate metrics -------------------------------- ###
    metrics_str = formatter.format(
        "Calculating test metrics",
        color="blue", 
        style=["bold"], 
        separator=True
    )
    print(metrics_str)
    
    # Basic metrics
    test_accuracy = np.mean(all_predictions == all_labels)
    avg_test_loss = np.mean(test_losses)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Classification report
    precision, recall, f1_score, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )
    
    # Macro and weighted averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    
    # F2 score (beta=2 gives more weight to recall)
    from sklearn.metrics import fbeta_score
    f2_macro = fbeta_score(all_labels, all_predictions, beta=2.0, average='macro', zero_division=0)
    f2_weighted = fbeta_score(all_labels, all_predictions, beta=2.0, average='weighted', zero_division=0)
    
    # ROC curves and AUC (for multiclass)
    if args.num_classes > 2:
        # Binarize labels for multiclass ROC
        y_test_bin = label_binarize(all_labels, classes=range(args.num_classes))
        
        # Compute ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(args.num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], all_probabilities[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and AUC
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), all_probabilities.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Compute macro-average ROC curve and AUC
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(args.num_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(args.num_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= args.num_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    else:
        # Binary classification
        fpr_bin, tpr_bin, _ = roc_curve(all_labels, all_probabilities[:, 1])
        roc_auc_bin = auc(fpr_bin, tpr_bin)
    
    print(f"[LOG] Test completed in {test_time:.2f} seconds")
    print(f"[LOG] Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"[LOG] Average Test Loss: {avg_test_loss:.4f}")
    print(f"[LOG] Macro F1-Score: {f1_macro:.4f}")
    print(f"[LOG] Weighted F1-Score: {f1_weighted:.4f}")
    print(f"[LOG] Gamma OOD: {args.gamma_ood}")
    print(f"[LOG] OH: {args.oh:.4f}")
    print(f"[LOG] ACT: {args.act:.4f}")
    print(f"[LOG] IE: {args.ie:.4f}")
    print(f"[LOG] G LR: {args.lr_G:.4f}")
    
    # Prepare metrics dictionary
    metrics = {
        'test_accuracy': test_accuracy,
        'avg_test_loss': avg_test_loss,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'f2_macro': f2_macro,
        'f2_weighted': f2_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'support': support,
        'confusion_matrix': cm,
        'test_time': test_time
    }
    
    if args.num_classes > 2:
        metrics.update({
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc
        })
    else:
        metrics.update({
            'fpr_bin': fpr_bin,
            'tpr_bin': tpr_bin,
            'roc_auc_bin': roc_auc_bin
        })

    ### -------------------------------- Generate PDF Report -------------------------------- ###
    if "_" in args.network:
        generate_test_report_pdf(
            args = args,
            metrics = metrics,
            model_name = model_name,
            dataset = args.dataset,
            class_names = class_names,
            t_model_name = teacher_name,
            approach_suffix = approach_suffix,
            student_test=True
        )
    else:
        generate_test_report_pdf(
            args = args,
            metrics = metrics,
            model_name = model_name,
            dataset = args.dataset,
            class_names = class_names,
            t_model_name = "",
            approach_suffix = "",
            student_test = False
        )
    
    final_str = formatter.format(
        f"Testing of {model_name} completed",
        color="green", 
        style=["bold"], 
        separator=True
    )
    print(final_str)