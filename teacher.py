import glob
import torch.nn as nn
import torch.optim as optim
import warnings
import time

from tqdm import tqdm

from datasets.dataloaders import init_train_test_loader
from networks.t_networks import get_teacher_model
from utils import *

warnings.filterwarnings("ignore", category=UserWarning, module='torchvision')
warnings.filterwarnings("ignore", category=UserWarning, module='torch.nn')

def train(args):
    ### ---------------------------------- Dataset loading ---------------------------------- ###
    torch.cuda.empty_cache()
    formatter = TextFormatter()
    data_str = formatter.format(
        f"Initializing dataset {args.dataset}",
        color = "blue", 
        style = ["bold"], 
        separator = True
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
    print(f"[LOG] Train dataset size: {len(train_dataset)}")
    print(f"[LOG] Test dataset size: {len(test_dataset)}")

    print_examples(args,train_loader,test_loader)
    print(f"[LOG] Saved sample examples in {args.save_path}")
        
    ### -------------------------------- Model initialization -------------------------------- ###
    model_str = formatter.format(
        "Initializing model",
        color = "blue", 
        style = ["bold"], 
        separator = True
    )
    print(model_str)

    t_model, t_model_name = get_teacher_model(args.t_network, args.num_classes)
    t_model.to(args.gpu)
    print(f"[LOG] Teacher model: {t_model_name}")

    ### -------------------------------- Train initialization -------------------------------- ###
    init_t_str = formatter.format(
        "Initializing training",
        color = "blue", 
        style = ["bold"], 
        separator = True
    )
    print(init_t_str)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(t_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=0.1,
        patience=args.patience,
        threshold=1e-4,
        verbose=True
    )

    print(f"[LOG] Loss function: {criterion.__class__.__name__}")
    print(f"[LOG] Optimizer: {optimizer.__class__.__name__}")
    print(f"[LOG] Learning rate scheduler: {scheduler.__class__.__name__}")
    print(f"[LOG] Total epochs: {args.epochs}")
    print(f"[LOG] Initial learning rate: {args.lr}")
    print(f"[LOG] Momentum: {args.momentum}")
    print(f"[LOG] Weight decay: {args.weight_decay}")
    print(f"[LOG] Patience: {args.patience}")

    best_acc = 0.0
    current_lr = args.lr
    patience = args.patience
    patience_counter = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }

    if not os.path.exists(f"{args.save_path}{args.dataset}/"):
        os.makedirs(f"{args.save_path}{args.dataset}/")
    
    ### ------------------------------------- Train loop ------------------------------------- ###
    init_t_str = formatter.format(
        f"Starting training of {t_model_name}",
        color = "blue", 
        style = ["bold"], 
        separator = True
    )
    print(init_t_str)

    start_time = time.time()
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")

        # Set model to training mode
        t_model.train()
        train_bar = tqdm(train_loader, desc="Training", leave=True)
        for batch_idx, (images, labels, _) in enumerate(train_bar):
            images = images.to(args.gpu)
            labels = labels.to(args.gpu)

            # Forward pass
            outputs = t_model(images)
            loss = criterion(outputs, labels)

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            accuracy = correct / labels.size(0)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update history
            history["train_loss"].append(loss.item())
            history["train_accuracy"].append(accuracy)

            # Update progress bar
            if (batch_idx + 1) % args.log_interval == 0:
                train_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{accuracy:.4f}", lr = f"{current_lr:.4f}")
        
        t_model.eval()  # Set model to evaluation mode
        test_bar = tqdm(test_loader, desc="Validation", leave=True)
        val_correct_total = 0
        val_samples_total = 0
        with torch.no_grad():
            for batch_idx, (images, labels, _) in enumerate(test_bar):
                images = images.to(args.gpu)
                labels = labels.to(args.gpu)

                # Forward pass
                outputs = t_model(images)
                loss = criterion(outputs, labels)

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                accuracy = correct / labels.size(0)
                val_correct_total += correct
                val_samples_total += labels.size(0)

                # Update history
                history["val_loss"].append(loss.item())
                history["val_accuracy"].append(accuracy)

                # Update progress bar
                if (batch_idx + 1) % args.log_interval == 0:
                    test_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{accuracy:.4f}")

        # Save the best model based on validation accuracy
        val_accuracy = val_correct_total / val_samples_total
        if val_accuracy > best_acc:
            for f in glob.glob(f"{args.save_path}{args.dataset}/{t_model_name.lower()}_best_*"):
                if os.path.isfile(f):
                    os.remove(f)

            best_acc = val_accuracy
            torch.save(t_model.state_dict(), f"{args.save_path}{args.dataset}/{t_model_name.lower()}_best_{epoch+1}.pth")
            
            save_str = formatter.format(
                f"New best model saved with accuracy: {best_acc:.4f}",
                color = "green", 
                style = ["underline"], 
                separator = True
            )
            print(save_str)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"[LOG] No improvement in accuracy. Patience counter: {patience_counter}/{patience}")

            if patience_counter > args.patience:
                early_str = formatter.format(
                    f"[EARLY STOPPING] No improvement for {patience} epochs. Stopping training.",
                    color = "green", 
                    style = ["underline"], 
                    separator = True
                )
                print(early_str)
                break

        # Update learning rate scheduler
        scheduler.step(val_accuracy)
        current_lr = optimizer.param_groups[0]['lr']

        # Tempo trascorso per questa epoca
        epoch_duration = time.time() - epoch_start_time

        # Tempo medio stimato per epoca finora
        epochs_done = epoch + 1
        avg_epoch_time = (time.time() - start_time) / epochs_done

        # ETA (in secondi)
        epochs_left = args.epochs - epochs_done
        eta_seconds = int(avg_epoch_time * epochs_left)

        # Formattazione leggibile
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
        early_str = formatter.format(
            f"Estimated time remaining: {eta_str}",
            color = "blue", 
            style = ["underline"], 
            separator = True
        )
        print(early_str)

    generate_training_report_pdf(args, history, t_model_name, best_acc)
    send_email(t_model_name, args.dataset, best_acc)

    final_str = formatter.format(
        f"Training of {t_model_name} completed",
        color = "green", 
        style = ["bold"], 
        separator = True
    )
    print(final_str)