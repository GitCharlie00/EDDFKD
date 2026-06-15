import glob
import time
import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets_local.dataloaders import init_train_test_loader
from networks.s_networks import get_student_model
from networks.t_networks import get_teacher_model
from sampling.DFND.dfnd import DFND
from sampling.evaluator import evaluator, save_best_model
from sampling.generator import Generator, PatchDiscriminator, generate_samples
from sampling.MOSAICK.mosaick import MOSAICK
from utils import TextFormatter, generate_gen_training_report_pdf, send_email
from sampling.utils_sam import DataIter, collect_sample_images, collect_sample_images_from_test

def dfkd_sam(args):
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

    if args.method.lower() == "mosaick":
        _, test_loader, _, _ = init_train_test_loader(
            dataset_type=args.dataset,
            dataset_root=args.dataset_root,
            train_batch=args.train_batch_size,
            test_batch=args.test_batch_size,
            num_workers=args.num_workers, 
        )

        train_loader, _, _, _ = init_train_test_loader(
            dataset_type=args.ood_dataset,
            dataset_root=args.dataset_root,
            train_batch=args.train_batch_size,
            test_batch=args.test_batch_size,
            num_workers=args.num_workers, 
            mosaick_ood=False
        )

        ood_loader, _, _, _ = init_train_test_loader(
            dataset_type=args.ood_dataset,
            dataset_root=args.dataset_root,
            train_batch=args.train_batch_size,
            test_batch=args.test_batch_size,
            num_workers=args.num_workers,
            mosaick_ood=True
        )
        ood_iter = DataIter(ood_loader)

        print(f"[LOG] Dataset type: {args.dataset}")
        print(f"[LOG] Validation dataset size: {len(test_loader.dataset)}")
        print(f"[LOG] Train dataset size: {len(train_loader.dataset)}")
        print(f"[LOG] OOD dataset size: {len(ood_loader.dataset)}")
    
    elif args.method.lower() == "dfnd":
        _, test_loader, _, _, teacher_acc = init_train_test_loader(
            dataset_type=args.dataset,
            dataset_root=args.dataset_root,
            train_batch=args.train_batch_size,
            test_batch=args.test_batch_size,
            num_workers=args.num_workers, 
            dfnd_acc=True
        )

        ood_train_loader, _, ood_train_dataset, _ = init_train_test_loader(
            dataset_type=args.ood_dataset,
            dataset_root=args.dataset_root,
            train_batch=args.train_batch_size,
            test_batch=args.test_batch_size,
            num_workers=args.num_workers, 
        )

        print(f"[LOG] Dataset type: {args.dataset}")
        print(f"[LOG] Validation dataset size: {len(test_loader.dataset)}")
        print(f"[LOG] OOD dataset size: {len(ood_train_loader.dataset)}")
        
    ### ----------------------------------- Teacher model ------------------------------------ ###
    teacher, teacher_name = get_teacher_model(args.t_network, args.num_classes, args.feature_extr)

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

    teacher.load_state_dict(torch.load(checkpoint_path, map_location=args.gpu))
    teacher.to(args.gpu)
    teacher.eval()
    print(f"[LOG] Teacher model successfully loaded from {matches[0]}")

    ### ----------------------------------- Student model ------------------------------------ ###
    student, student_name = get_student_model(args.s_network, args.num_classes)

    stud_str = formatter.format(
        f"Initializing student network {student_name}",
        color = "blue", 
        style = ["bold"], 
        separator = True
    )
    print(stud_str)

    student.to(args.gpu)
    print(f"[LOG] Student model successfully initialized")

    ### ---------------------------------- Generator model ----------------------------------- ### 
    if args.method.lower() == "mosaick":
        gen_str = formatter.format(
            f"Initializing generator-discriminator networks",
            color = "blue", 
            style = ["bold"], 
            separator = True
        )
        print(gen_str)

        G = Generator(
            nz=args.z_dim,
            nc=args.nc,
            img_size=args.img_size 
        ).to(args.gpu)

        D = PatchDiscriminator(
            nc=args.nc, 
            ndf=args.ndf
        ).to(args.gpu)

        print(f"[LOG] Generator-Discrimator models successfully initialized")
    elif args.method.lower() == "dfnd":
        print(f"[LOG] Generator not necessary")

    ### ------------------------------ Generated Dataset Setup ------------------------------- ###
    if args.method.lower() == "mosaick":
        dataset_save_dir = f"{args.dataset_root}{teacher_name}_{args.method.lower()}/"    
        print(f"[LOG] Generated dataset will be saved to: {dataset_save_dir}")
    
    ### -------------------------------- Train initialization -------------------------------- ###
    init_str = formatter.format(
        f"Initializing {args.method.upper()} training",
        color = "blue", 
        style = ["bold"], 
        separator = True
    )
    print(init_str)

    history = {
        "S_train_loss": [],
        "S_train_accuracy": [],
        "S_val_loss": [],
        "S_val_accuracy": []
    }

    opt_S = optim.SGD(
        student.parameters(),
        lr=args.lr_S,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    val_criterion = nn.CrossEntropyLoss()

    if args.method.lower() == "mosaick":
        opt_G = optim.Adam(
            G.parameters(), 
            lr=args.lr_G, 
            betas=[args.beta_1, args.beta_2]
        )
        scheduler_G = CosineAnnealingLR(opt_G, T_max=args.epochs*len(train_loader.dataset))

        print(f"[LOG] Generator optimizer: {opt_G.__class__.__name__}")
        print(f"[LOG] Generator learning rate: {args.lr_G}")

        opt_D = optim.Adam(
            D.parameters(), 
            lr=args.lr_G, 
            betas=[args.beta_1, args.beta_2]
        )
        scheduler_D = CosineAnnealingLR(opt_D, T_max=args.epochs*len(train_loader.dataset))

        print(f"[LOG] Discriminator optimizer: {opt_D.__class__.__name__}")
        print(f"[LOG] Discriminator learning rate: {args.lr_G}")

        history["G_train_loss"] = []
        history["D_train_loss"] = []
        history["G_LOC_train_loss"]  = {"values":[],"name":"Local"} # Fool the patch discriminator
        history["G_ALG_train_loss"]  = {"values":[],"name":"Align"} # Label space aligning
        history["G_ADV_train_loss"]  = {"values":[],"name":"Info Entropy"} # Fool the student

        method_labels = [f"L_{key[2:-11]}" for key in history.keys() if key.startswith("G_") and key != "G_train_loss"]

        scheduler_S = CosineAnnealingLR(opt_S, T_max=args.epochs*len(train_loader.dataset))
    
    elif args.method.lower() == "dfnd":
        args.epochs = 1 #int(40000/args.num_select * 512)
        history["N_total_loss"] = []
        history["N_KDL_loss"]   = {"values":[],"name":"KL-div"} # Minimize distance between teacher and student logits
        history["N_CE_loss"]    = {"values":[],"name":"Cross-entropy"} # CE to handle noisy labels

        method_labels = [f"L_{history[k]['name']}" for k in history if k.startswith("N_") and k != "N_total_loss"]

        scheduler_S = CosineAnnealingLR(opt_S, 200, eta_min=2e-4)

    print(f"[LOG] Student optimizer: {opt_S.__class__.__name__}")
    print(f"[LOG] Student learning rate: {args.lr_S}")

    print(f"[LOG] Total epochs: {args.epochs}")
    
    best_acc = 0.0

    ### -------------------------------- Method configuration -------------------------------- ###         
    if args.method.lower() == "mosaick":
        method_cfg = dict(
            args = args,
            models = [teacher,student,G,D],
            optims = [opt_S,opt_G,opt_D],
            schedulers = [scheduler_S,scheduler_G,scheduler_D],
            loaders = [train_loader, ood_iter],
            history = history,
            method_labels = method_labels
        )
        dfkd_method = MOSAICK(**method_cfg)

    elif args.method.lower() == "dfnd":
        method_cfg = dict(
            args = args,
            ood_train_dataset = ood_train_dataset,
            teacher = teacher,
            student = student,
            scheduler_S = scheduler_S,
            opt_S = opt_S,
            history = history,
            teacher_acc = teacher_acc,
            method_labels = method_labels
        )
        dfkd_method = DFND(**method_cfg)
    
    ### ------------------------------------ Train loop ------------------------------------- ###
    start_str = formatter.format(
        f"Starting {args.method.upper()} training",
        color = "blue", 
        style = ["bold"], 
        separator = True
    )
    print(start_str)

    start_time = time.time()
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")

        # Method train
        dfkd_method.train_loop()
        
        # Student evaluation
        val_correct_total, val_samples_total = evaluator(student,test_loader,val_criterion,history,args.gpu,args.log_interval)

        # Save the best model based on validation accuracy
        val_accuracy = val_correct_total / val_samples_total
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            save_best_model(args,teacher_name,student_name,student,epoch)
            save_str = formatter.format(
                f"New best student saved with accuracy: {best_acc:.4f}",
                color = "green", 
                style = ["underline"], 
                separator = True
            )
            print(save_str)
        
        # ETA computation
        epoch_duration = time.time() - epoch_start_time
        epochs_done = epoch + 1
        avg_epoch_time = (time.time() - start_time) / epochs_done
        epochs_left = args.epochs - epochs_done
        eta_seconds = int(avg_epoch_time * epochs_left)

        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
        early_str = formatter.format(
            f"Estimated time remaining: {eta_str}\n",
            color = "blue", 
            style = ["underline"], 
            separator = True
        )
        print(early_str)

    ### ---------------------------- Generate and Save Dataset ----------------------------- ###
    if args.method.lower() == "mosaick":
        total_images_to_generate = args.dataset_size // 50
        save_dataset_str = formatter.format(
            f"Generating and saving dataset with {total_images_to_generate} images",
            color = "yellow", 
            style = ["bold"], 
            separator = True
        )
        print(save_dataset_str)
        
        label_count, image_counter = generate_samples(args,G,teacher,dataset_save_dir,total_images_to_generate)
        
        dataset_saved_str = formatter.format(
            f"Generated dataset saved successfully to {dataset_save_dir}",
            color = "green", 
            style = ["unrderline"], 
            separator = True
        )
        print(dataset_saved_str)
        print(f"[LOG] Total images saved: {image_counter}")
    
    ### ---------------------------- Training Report and Email ----------------------------- ###    
    if args.method.lower() == "mosaick":
        sample_images = collect_sample_images(dataset_save_dir, num_samples=10)
    else:
        dataset_save_dir = f"{args.dataset_root}{teacher_name}_{args.method.lower()}/"    
        sample_images, label_count = collect_sample_images_from_test(test_loader, teacher, args.gpu, num_samples=10, save_dir=dataset_save_dir, dataset=args.dataset)

    generate_gen_training_report_pdf(args, history, teacher_name, student_name, args.dataset, best_acc, sample_images, label_count)
    send_email(student_name, args.dataset, best_acc, args.approach, args.method)

    final_str = formatter.format(
        f"{args.method.upper()} training of {student_name} completed",
        color = "green", 
        style = ["bold"], 
        separator = True
    )
    print(final_str)