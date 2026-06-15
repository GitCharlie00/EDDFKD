import glob
import time
import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from tqdm import tqdm

from generative.DAFL.dafl import DAFL
from generative.DEEPINV.deepinv import DEEPINV
from datasets_local.dataloaders import init_train_test_loader
from generative.CMI.cmi import CMI
from generative.FAST.fast import FAST
from generative.generator import get_generator, generate_samples, generate_samples_deepinv
from networks.s_networks import get_student_model
from networks.t_networks import get_teacher_model
from plot import plot
from test import test
from utils import approach_label, generate_gen_training_report_pdf, get_gpu_memory_usage, save_result_json, send_email, TextFormatter
from generative.utils_gen import apply_datafree, collect_sample_images
from generative.evaluator import evaluator, save_best_model

def dfkd_gen(args):
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
    _, test_loader, train_dataset, test_dataset = init_train_test_loader(
        dataset_type=args.dataset,
        dataset_root=args.dataset_root,
        train_batch=args.train_batch_size,
        test_batch=args.test_batch_size,
        num_workers=args.num_workers
    )

    print(f"[LOG] Dataset type: {args.dataset}")
    print(f"[LOG] Validation dataset size: {len(test_dataset)}")
    print(f"[LOG] Original dataset size: {len(train_dataset)}")
        
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

    try:
        teacher.load_state_dict(torch.load(checkpoint_path, map_location=args.gpu)["state_dict"])
    except:
        teacher.load_state_dict(torch.load(checkpoint_path, map_location=args.gpu))
    
    # teacher.load_state_dict(torch.load(checkpoint_path, map_location=args.gpu), strict=False)
    teacher.to(args.gpu)
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
    if args.method.lower() != "deepinv":
        gen_str = formatter.format(
            f"Initializing generator network",
            color = "blue", 
            style = ["bold"], 
            separator = True
        )
        print(gen_str)

        G = get_generator(nz=args.z_dim,method=args.method.lower()).to(args.gpu)
        print(f"[LOG] Generator model successfully initialized")
    else:
        G = None
        print(f"[LOG] Generator model not required")
    
    ### ------------------------------ Generated Dataset Setup ------------------------------- ###
    approach_suffix = approach_label(args)
    user_suffix = f"_{args.suffix}" if args.suffix else ""
    dataset_save_dir = f"{args.dataset_root}{teacher_name}_{student_name}_{args.method.lower()}{approach_suffix}_{args.dataset}{user_suffix}/"

    print(f"[LOG] Generated dataset will be saved to: {dataset_save_dir}")

    ### -------------------------------- Train initialization -------------------------------- ###
    init_str = formatter.format(
        f"Initializing {args.method.upper()} training",
        color = "blue", 
        style = ["bold"], 
        separator = True
    )
    print(init_str)

    print(f"[LOG] Generator optimizer will be instantiated later")

    if args.kdci:
        Wq = nn.Linear(args.num_classes, args.hidden_dim, bias=False)
        Wk = nn.Linear(args.num_classes, args.hidden_dim, bias=False)
        Wt = nn.Linear(args.hidden_dim, 1)
        W = [Wq,Wk,Wt]

        opt_S = optim.SGD(
            list(student.parameters()) + list(Wq.parameters()) + list(Wk.parameters()) + list(Wt.parameters()),
            lr=args.lr_S,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    else:
        opt_S = optim.SGD(
            student.parameters(),
            lr=args.lr_S,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )

    print(f"[LOG] Student optimizer: {opt_S.__class__.__name__}")
    print(f"[LOG] Student learning rate: {args.lr_S}")

    val_criterion = nn.CrossEntropyLoss()
    if args.method == "dafl":
        if args.dataset == "cifar100" and "vgg" in args.t_network:
            scheduler_S = MultiStepLR(opt_S, milestones=[int(args.epochs*0.2), int(args.epochs*0.6)], gamma=0.1)
        else:
            scheduler_S = MultiStepLR(opt_S, milestones=[int(args.epochs*0.4), int(args.epochs*0.8)], gamma=0.1)
    else:
        scheduler_S = CosineAnnealingLR(opt_S, args.epochs, eta_min=2e-4)

    steps_per_epoch = args.ep_steps // args.kd_steps
    print(f"[LOG] Total epochs: {args.epochs}")
    print(f"[LOG] Steps per epochs: {steps_per_epoch}")
    
    history = {
        "G_train_loss": [],
        "S_train_loss": [],
        "S_train_accuracy": [],
        "S_val_loss": [],
        "S_val_accuracy": []
    }
    
    if args.method == "dafl":
        history["G_OH_train_loss"]   = {"values":[],"name":"One-Hot"} # Encourages the outputs of generated images by the teacher network to be close to one-hot like vectors
        history["G_ACT_train_loss"]  = {"values":[],"name":"Activations"} # Feature maps tend to receive higher activation value if input images are real rather than some random vectors
        history["G_IE_train_loss"]   = {"values":[],"name":"Info Entropy"} # How much G can generate images of each category with roughly the same probability
    elif args.method == "fast":
        history["G_OH_train_loss"]   = {"values":[],"name":"One-Hot"} # Encourages the outputs of generated images by the teacher network to be close to one-hot like vectors
        history["G_ADV_train_loss"]  = {"values":[],"name":"Adversarial"} # Improve the generation of synthetic data by aligning the probability distributions generated by the teacher model and the student model
        history["G_FEAT_train_loss"] = {"values":[],"name":"Features"} # Aims to align the teacher internal process
    elif args.method == "cmi":
        history["G_OH_train_loss"]   = {"values":[],"name":"One-Hot"} # Encourages the outputs of generated images by the teacher network to be close to one-hot like vectors
        history["G_BN_train_loss"]   = {"values":[],"name":"Regularization"} # Aims to align the batch normalisation statistics between the synthetic data generated and the original data used by the teacher model
        history["G_ADV_train_loss"]  = {"values":[],"name":"Adversarial"} # Improve the generation of synthetic data by aligning the probability distributions generated by the teacher model and the student model
        history["G_CR_train_loss"]   = {"values":[],"name":"Contrastive"} # How to distinguish different samples by push positive pairs closer and pull negative pairs apart
    elif args.method == "deepinv":
        history["G_OH_train_loss"]   = {"values":[],"name":"One-Hot"} # Encourages the outputs of generated images by the teacher network to be close to one-hot like vectors
        history["G_BN_train_loss"]   = {"values":[],"name":"Regularization"} # Aims to align the batch normalisation statistics between the synthetic data generated and the original data used by the teacher model
        history["G_ADV_train_loss"]  = {"values":[],"name":"Adversarial"} # Improve the generation of synthetic data by aligning the probability distributions generated by the teacher model and the student model
        history["G_TV_train_loss"]   = {"values":[],"name":"Total Variation"} # Improve the generation of realistic synthetic images

    history["G_E_loss"]       = {"values":[],"name":"Generator Energy loss"} # Shape the genrated distribution similarly to the original one
    
    method_labels = ["L_G"]
    method_labels += ["L_" + str(key.split("_")[1]) for key in list(history.keys())[5:]]
    method_labels += ["_KD_".join([key.split("_")[0], key.split("_")[2]]) for key in list(history.keys())[1:3]]
    
    history["G_lr_values"]    = {"values":[],"name":"Student LR values"} # Shape the genrated distribution similarly to the original one
    history["G_diversity"]    = {"values":[],"name":"Generator diversity"} # Diversity between the generator synthetized samples
    history["G_T_entropy"]    = {"values":[],"name":"Teacher entropy on G running"} # Teacher entropy on generated samples while G is learning

    history["T_S_agreement"]  = {"values":[],"name":"Prediction agreement"} # Prediction agreement between T and S
    history["T_S_energy"]     = {"values":[],"name":"Teacher energy during distillation"} # Teacher energy on generated samples while G is fixed
    history["S_energy"]       = {"values":[],"name":"Student energy during distillation"} # S energy
    history["T_S_match"]      = {"values":[],"name":"T-S energy agreement"} # Energy agreement between T and S
    history["T_S_entropy"]    = {"values":[],"name":"Teacher entropy during distillation"} # Teacher entropy on generated samples while G is fixed


    ### -------------------------------- Method configuration -------------------------------- ###         
    best_acc = 0.0
    warmup_printed = False
    transform = train_dataset.transform if not args.method == "dafl" else None 

    method_cfg = dict(
        args = args,
        teacher = teacher,
        student = student,
        G = G,
        history = history,
        opt_S = opt_S,
        W = W if args.kdci else None,
        transform = transform,
        save_dir = dataset_save_dir
    )
    
    if args.method == "dafl":                 
        dfkd_method = DAFL(**method_cfg)
    elif args.method == "fast":
        dfkd_method = FAST(**method_cfg)
    elif args.method == "cmi":
        dfkd_method = CMI(**method_cfg)
    elif args.method == "deepinv":
        dfkd_method = DEEPINV(**method_cfg)

    ### ------------------------------------ Train loop ------------------------------------- ###
    if args.gpu is not None:
        torch.cuda.reset_peak_memory_stats(device=args.gpu)

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

        train_bar = tqdm(range(steps_per_epoch), desc="Training", leave=True)
        for step in train_bar:
            G_losses, S_losses, warmup_printed = apply_datafree(args,epoch,dfkd_method,warmup_printed)

            if (step + 1) % args.log_interval == 0:
                # Update tqdm
                postfix_dict = {}
                losses = G_losses + S_losses

                for i, label in enumerate(method_labels):
                    postfix_dict[label] = losses[i].item()
                
                train_bar.set_postfix(postfix_dict)
        
        if epoch == 0 and args.gpu is not None:
            peak_mem = torch.cuda.max_memory_allocated(device=args.gpu) / (1024 ** 2) # Converte in MB
            mem_str = formatter.format(
                f"Peak GPU Memory Usage: {peak_mem:.2f} MB",
                color = "cyan", 
                style = ["bold"], 
                separator = True
            )
            print(mem_str)

        # Student evaluation
        val_correct_total, val_samples_total = evaluator(student,test_loader,val_criterion,history,args.gpu,args.log_interval)
        
        # Save the best model based on validation accuracy
        val_accuracy = val_correct_total / val_samples_total
        print(f"[LOG] Validation accuracy: {val_accuracy:.4f}")
        print(f"[LOG] Best accuracy: {best_acc:.4f}")
        print(f"[LOG] Distillation: {teacher_name} -> {student_name}")
        print(f"[LOG] Dataset: {args.dataset}")
        
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            model_name_save_path = save_best_model(args,teacher_name,student_name,student,approach_suffix+user_suffix,epoch)
            save_str = formatter.format(
                f"New best student saved with accuracy: {best_acc:.4f}",
                color = "green", 
                style = ["underline"], 
                separator = True
            )
            print(save_str)

        # Update learning rate
        if args.method != "fast" or epoch >= args.warmup:
            scheduler_S.step()
        
        # ETA computation
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

    total_training_time = time.time() - start_time
    training_time_str = time.strftime("%H:%M:%S", time.gmtime(int(total_training_time)))
    
    training_complete_str = formatter.format(
        f"Training completed in {training_time_str} (Total: {total_training_time:.2f} seconds)",
        color = "green", 
        style = ["bold"], 
        separator = True
    )
    print(training_complete_str)

    ### ---------------------------- Generate and Save Dataset ----------------------------- ###
    total_images_to_generate = args.dataset_size // 50
    save_dataset_str = formatter.format(
        f"Generating and saving dataset with {total_images_to_generate} images",
        color = "yellow", 
        style = ["bold"], 
        separator = True
    )
    print(save_dataset_str)
    
    if args.method.lower() != "deepinv":
        label_count, image_counter = generate_samples(args,G,teacher,dataset_save_dir,total_images_to_generate)
    else:
        label_count, image_counter = generate_samples_deepinv(
            args = args,
            teacher = teacher,
            dataset_save_dir = dataset_save_dir,
            total_images_to_generate = total_images_to_generate,
            data_pool = dfkd_method.get_data_pool()
        )
    
    dataset_saved_str = formatter.format(
        f"Generated dataset saved successfully to {dataset_save_dir}",
        color = "green", 
        style = ["unrderline"], 
        separator = True
    )
    print(dataset_saved_str)
    print(f"[LOG] Total images saved: {image_counter}")
    
    ### ---------------------------- Training Report and Email ----------------------------- ###    
    sample_images = collect_sample_images(dataset_save_dir, num_samples=10)
    generate_gen_training_report_pdf(args, history, teacher_name, student_name, args.dataset, best_acc, sample_images, label_count, approach_suffix+user_suffix)
    
    save_result_json(args,args.dataset, args.method, approach_suffix+user_suffix, teacher_name, student_name, best_acc, str(total_training_time), training_time_str, memory_usage=f"{peak_mem:.2f}M",results_dir="work/project/results/",test=False)
    plot(args,approach_suffix,user_suffix)

    args.network = args.s_network
    test(args,user_suffix)

    send_email(
        network=student_name, 
        dataset=args.dataset, 
        best_acc=best_acc, 
        approach=args.approach, 
        method= " ".join(item.upper() for item in (approach_suffix+user_suffix).split("_")),
        teacher=teacher_name
    )

    final_str = formatter.format(
        f"{args.method.upper()} training of {student_name} completed",
        color = "green", 
        style = ["bold"], 
        separator = True
    )
    print(final_str)