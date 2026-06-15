import argparse
import os
import sys

# ANSI escape codes for colors and styles
RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
BLUE   = "\033[34m"

def print_log(message: str):
    print(f"{BLUE}[LOG]{RESET} {message}")

def print_error(message: str):
    print(f"{BOLD}{RED}[ERROR]{RESET} {message}")

def print_success(message: str):
    print(f"{BOLD}{GREEN}{message}{RESET}")

def main():
    # Definizione esplicita delle liste di metodi
    GENERATIVE_METHODS = ["DAFL", "Fast", "CMI", "DeepInv"]
    SAMPLING_METHODS  = ["Mosaick", "DFND"]
    ALL_METHODS       = GENERATIVE_METHODS + SAMPLING_METHODS

    parser = argparse.ArgumentParser(
        description="Create a config file for train, test or DFKD."
    )
    
    parser.add_argument(
        "main_script",
        type=str,
        help="Path to the main script."
    )

    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the config file to generate (.txt)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "dfkd", "test", "plot"],
        help="Mode: train / dfkd / test / plot."
    )
    
    # Common arguments
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "cifar100", "svhn", "tiny_imagenet", "sun", "places365", "imagenet"],
        required=True,
        help="Dataset name."
    )
    parser.add_argument(
        "--network",
        type=str,
        help="Network for train/test mode."
    )
    parser.add_argument(
        "--t_network",
        type=str,
        help="Teacher network (DFKD)."
    )
    parser.add_argument(
        "--s_network",
        type=str,
        help="Student network (DFKD)."
    )
    parser.add_argument(
        "--approach",
        type=str,
        choices=["generative", "sampling"],
        help="DFKD approach."
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=ALL_METHODS,
        help="DFKD method (dependent on --approach)."
    )
    parser.add_argument(
        "--kdci",
        type=str,
        choices=["true", "false"],
        help="Integrate KDCI? Set to 'true' or 'false'."
    )

    parser.add_argument(
        "--ood_loss",
        type=str,
        choices=["true", "false"],
        help="Integrate OOD loss? Set to 'true' or 'false'."
    )

    parser.add_argument(
        "--additive_loss",
        type=str,
        choices=["true", "false"],
        help="Should energy loss term be additive? Set to 'true' or 'false'."
    )

    parser.add_argument(
        "--gamma_ood",
        type=float,
        help="OOD gamma value (0.1, 0.3, 0.5, 0.7, 1.0)"
    )

    parser.add_argument(
        "--lr_G",
        type=float,
        help="Generator LR value"
    )

    parser.add_argument(
        "--oh",
        type=float,
        help="OH value"
    )

    parser.add_argument(
        "--act",
        type=float,
        help="ACT value"
    )

    parser.add_argument(
        "--ie",
        type=float,
        help="IE value"
    )

    parser.add_argument(
        "--adv",
        type=float,
        help="ADV value"
    )

    parser.add_argument(
        "--feat",
        type=float,
        help="FEAT value"
    )

    parser.add_argument(
        "--gamma_adaptive",
        type=str,
        help="Adaptive gamma value 'single' or 'double'"
    )

    parser.add_argument(
        "--new_gamma_ood",
        type=float,
        help="New OOD gamma value (0.1, 0.3, 0.5, 0.7, 1.0)"
    )

    parser.add_argument(
        "--g_reset",
        type=str,
        choices=["true", "false"],
        help="Integrate G reset? Set to 'true' or 'false'."
    )

    parser.add_argument(
        "--g_penality",
        type=str,
        choices=["true", "false"],
        help="Integrate G penality? Set to 'true' or 'false'."
    )
    
    parser.add_argument(
        "--s_energy_match",
        type=str,
        choices=["true", "false"],
        help="Integrate S-T energy match? Set to 'true' or 'false'."
    )

    parser.add_argument(
        "--calc_target",
        type=str,
        choices=["true", "false"],
        help="Integrate energy target computation? Set to 'true' or 'false'."
    )

    parser.add_argument(
        "--energy_kd",
        type=str,
        choices=["true", "false"],
        help="Integrate energy-weighted KD? Set to 'true' or 'false'."
    )

    parser.add_argument(
        "--energy_kd_beta",
        type=float,
        help="Energy-KD gate sharpness (default 1.0)"
    )

    parser.add_argument(
        "--gpu_id",
        type=int,
        help="GPU index"
    )

    parser.add_argument(
        "--suffix",
        type=str,
        help="User defined suffix"
    )

    args = parser.parse_args()

    # config file must end with .txt
    if not args.config_file.endswith(".txt"):
        print_error("Config file must end with .txt")
        sys.exit(1)

    mode = args.mode
    dataset = args.dataset
    calc_target = args.calc_target

    # Determine number of classes
    if dataset == "cifar10":
        num_classes = 10
    elif dataset == "cifar100":
        num_classes = 100
    elif dataset == "svhn":
        num_classes = 10
    elif dataset == "tiny_imagenet":
        num_classes = 200
    elif dataset == "sun":
        num_classes = 397
    elif dataset == "places365":
        num_classes = 365
    elif dataset == "imagenet":
        num_classes = 1000
    else:
        print_error("Unsupported dataset.")
        sys.exit(1)

    # Shared paths & hyperparams
    dataset_root = "/work/data/ssd_datasets/"
    save_path    = "/work/data/ssd_results/"
    log_interval = 10
    num_workers  = 8 #4
    gpu_id = args.gpu_id if args.gpu_id else 0

    # Optimization defaults
    if args.mode == "train":
        # if args.network == "resnet-34":
        if "resnet" in args.network:
            weight_decay = 5e-4
            lr           = 0.1
            momentum     = 0.9
            epochs       = 200
            train_batch  = 128
            test_batch   = 128
            
        if args.network == "vgg-11":
            weight_decay = 5e-4
            lr           = 0.1
            momentum     = 0.9
            epochs       = 200
            train_batch  = 128
            test_batch   = 128

        if "wrn" in args.network:
            weight_decay = 5e-4
            lr           = 0.1
            momentum     = 0.9
            epochs       = 200
            train_batch  = 128
            test_batch   = 128
            
        # --- SPECIALIZZAZIONE PER DATASET ---
        # SVHN converge velocemente, spesso bastano meno epoche, ma 200 va bene
        if dataset == "svhn":
            lr = 0.01 # SVHN spesso preferisce LR più basso
        
        # Dataset pesanti (224x224): Riduciamo batch size per evitare OOM su 3090
        if dataset in ["sun", "places365", "imagenet"]:
            train_batch = 64 
            test_batch  = 64
            # Opzionale: Places è enorme, potresti volere meno epoche o LR diverso
            if dataset == "places365":
                epochs = 100
                
        if dataset == "imagenet":
            train_batch = 64   # 128 potrebbe andare in OOM su 24GB con ResNet34/50
            test_batch = 64
            epochs = 100       # 90-100 epoche è standard per ImageNet
            lr = 0.1
            weight_decay = 1e-4 # Standard ImageNet setting
    
    elif (args.mode == "dfkd" or args.mode == "plot") and args.approach == "generative":
        dataset_size      = 50000
        temperature       = 20.0
        data_clear        = 10
        keep_last         = 2560
        hidden_dim        = 256
        in_energy_target  = -23.0 if args.dataset == "cifar10" else -27.0
        out_energy_target = -5.0
        
        # if "resnet" in args.t_network:
        #     if args.dataset == "cifar10":
        #         energy_target = -9.82
        #     else:
        #         energy_target = -10.50

        # if "vgg" in args.t_network:
        #     if args.dataset == "cifar10":
        #         energy_target = -8.71
        #     else:
        #         energy_target = -11.08

        # if "wrn" in args.t_network:
        #     if args.dataset == "cifar10":
        #         energy_target = -10.51
        #     else:
        #         energy_target = -11.99

        if args.method.lower() == "dafl":
            if args.dataset == "cifar10":
                train_batch_size = 1024 #256
                test_batch       = 1024 #100
                z_dim            = 1000 #512
                epochs           = 2000 #250
                lr_G             = args.lr_G if args.lr_G is not None else 0.002
                lr_S             = 0.1
                oh               = args.oh if args.oh is not None else 1.0 
                act              = args.act if args.act is not None else 0.001 
                ie               = args.ie if args.ie is not None else 20 
                kd_steps         = 1    #5
                ep_steps         = 120  #400
                # lr_G             = 0.002 #0.02 #1e-3 VGG
                # oh               = 0.5
                # act              = 0.1  #if args.ood_loss.lower() != "true" else 0.001
                # ie               = 20    #if args.ood_loss.lower() != "true" else 20

            else:
                train_batch_size = 1024
                test_batch       = 1024 #100
                z_dim            = 1000
                epochs           = 2000
                lr_G             = args.lr_G if args.lr_G is not None else 0.002
                lr_S             = 0.1
                oh               = args.oh if args.oh is not None else 1.0 
                act              = args.act if args.act is not None else 0.001 
                ie               = args.ie if args.ie is not None else 20 
                kd_steps         = 1      
                ep_steps         = 120 
                # lr_G             = 0.005 if "vgg" not in args.t_network else 2e-3
                # lr_S             = 0.1  if "vgg" not in args.t_network else 0.01    

            weight_decay     = 5e-4
            momentum         = 0.9
            feature_extr     = True
            g_steps          = 1
            confounder_size  = 8

            # if "vgg" in args.t_network:
            #     lr_S = 0.01          # Ridotto da 0.1
            #     weight_decay = 5e-3  # Aumentato da 5e-4
            #     oh = 0.3             # Ridotto da 0.5
            #     ie = 30.0            # Aumentato da 20.0

        # elif args.method.lower() == "fast":
        #     z_dim            = 256
        #     epochs           = 2000 #250
        #     lr_G             = 1e-3
        #     lr_S             = 0.1
        #     momentum         = 0.9
        #     feature_extr     = True
        #     kd_steps         = 2#400
        #     ep_steps         = 2#400
        #     g_steps          = 2#10
        #     oh               = 0.4
        #     adv              = 1.1
        #     feat             = 10.0
        #     warmup           = 20
        #     lr_z             = 0.01
        #     confounder_size  = 8
        elif args.method.lower() == "fast":
            train_batch_size = 256
            test_batch       = 256
            z_dim            = 256
            epochs           = 220
            lr_G             = args.lr_G if args.lr_G is not None else 0.002
            lr_S             = 0.2 #lr
            momentum         = 0.9
            feature_extr     = True
            kd_steps         = 400
            ep_steps         = 400
            g_steps          = 10
            oh               = args.oh if args.oh is not None else 0.4
            adv              = args.adv if args.adv is not None else 1.1 
            feat             = args.feat if args.feat is not None else 10 
            warmup           = 20
            lr_z             = 0.01
            confounder_size  = 8
            weight_decay     = 1e-4
            num_workers      = 4
            seed = 0
            # lr_G             = 2e-3
            # oh               = 0.5
            # adv              = 1.33
            # feat             = 10.0 #bn

        elif args.method.lower() == "cmi":
            z_dim            = 256
            epochs           = 250
            lr_G             = 1e-3
            lr_S             = 0.1
            feature_extr     = True
            kd_steps         = 400
            ep_steps         = 400
            g_steps          = 200
            beta_1           = 0.5
            beta_2           = 0.999
            adv              = 0.5
            bn               = 1.0
            oh               = 1.0
            cr               = 0.8
            cr_T             = 0.2
            confounder_size  = 32
        elif args.method.lower() == "deepinv":
            epochs           = 250
            lr_G             = 0.1
            lr_S             = 0.1
            feature_extr     = True
            kd_steps         = 400
            ep_steps         = 400
            g_steps          = 1000
            beta_1           = 0.5
            beta_2           = 0.99
            adv              = 1.0
            bn               = 10.0
            oh               = 1.0
            tv               = 1e-5
            confounder_size  = 32
    elif (args.mode == "dfkd" or args.mode == "plot") and args.approach == "sampling":
        train_batch_size = 256
        test_batch       = 100
        momentum         = 0.9
        hidden_dim       = 256

        if args.method.lower() == "mosaick":
            ood_dataset      = "cifar100" if dataset == "cifar10" else "cifar10"
            ood_classes      = 100 if num_classes == 10 else 10
            weight_decay     = 1e-4
            z_dim            = 100
            epochs           = 200
            lr_S             = 0.1
            lr_G             = 1e-3
            feature_extr     = False
            kd_steps         = 4
            align            = 0.1
            local            = 0.1
            adv              = 1.0
            nc               = 3
            img_size         = 32
            ndf              = 128
            beta_1           = 0.5
            beta_2           = 0.999
            temperature      = 1.0
            dataset_size     = 50000
            confounder_size  = 128

        elif args.method.lower() == "dfnd":
            ood_dataset      = "cifar100" if dataset == "cifar10" else "cifar10"
            weight_decay     = 5e-4
            num_select       = 40000
            epochs           = 50
            lr_S             = 0.1
            lr_N             = 1e-3
            feature_extr     = False
            confounder_size  = 128
        else:
            print_error("Invalid method.")
            sys.exit(1)

    elif mode == "test" or mode == "plot":
        test_batch = 256
    else:
        print_error("Invalid mode")
        sys.exit(1)

    # Start building config lines
    lines = [
        f"--mode {mode}\n",
        f"--gpu_id {gpu_id}\n"
        "\n"
    ]

    if mode == "plot":
        lines += [f"--calc_target\n" if args.calc_target.lower() == "true" else "\n"]

    if mode == "train":
        if not args.network:
            print_error("Train mode requires --network.")
            sys.exit(1)
        lines += [
            f"--dataset {dataset}\n",
            f"--dataset_root {dataset_root}\n",
            f"--num_classes {num_classes}\n",
            f"--num_workers {num_workers}\n",
            "\n",
            f"--save_path {save_path}\n",
            f"--log_interval {log_interval}\n",
            "\n",
            f"--network {args.network}\n",
            "\n",
            f"--lr {lr}\n",
            f"--momentum {momentum}\n",
            f"--weight_decay {weight_decay}\n",
            "\n",
            f"--train_batch_size {train_batch}\n",
            f"--epochs {epochs}\n",
            "\n",
            f"--test_batch_size {test_batch}\n"
        ]

    elif mode == "dfkd" or mode == "plot":

        # require teacher, student, approach, method and kdci
        if (not args.t_network or not args.s_network or not args.approach
                or not args.method or args.kdci is None):
            print_error("DFKD requires --t_network, --s_network, --approach, --method and --kdci.")
            sys.exit(1)

        # validate method vs. approach
        if args.approach == "generative":
            if args.method not in GENERATIVE_METHODS:
                print_error(f"Method '{args.method}' invalid for generative approach. Choose one of {GENERATIVE_METHODS}.")
                sys.exit(1)

            if args.method.lower() == "dafl":
                lines += [
                    f"--dataset {dataset}\n",
                    f"--dataset_root {dataset_root}\n",
                    f"--num_classes {num_classes}\n",
                    f"--num_workers {num_workers}\n",
                    "\n",
                    f"--save_path {save_path}\n",
                    f"--log_interval {log_interval}\n",
                    "\n",
                    f"--t_network {args.t_network}\n",
                    f"--s_network {args.s_network}\n",
                    "\n",
                    f"--approach {args.approach}\n",
                    f"--method {args.method.lower()}\n",
                    f"--kdci \n" if args.kdci.lower() == "true" else "",
                    f"--confounder_size {confounder_size}\n",
                    f"--hidden_dim {hidden_dim} \n", 
                    f"--ood_loss \n" if args.ood_loss.lower() == "true" else "",
                    f"--additive_loss \n" if (args.additive_loss and args.additive_loss.lower() == "true") else "",
                    f"--gamma_ood {args.gamma_ood}\n" if args.gamma_ood else "",
                    f"--gamma_adaptive {args.gamma_adaptive}\n",
                    f"--new_gamma_ood {args.new_gamma_ood}\n",
                    f"--g_reset \n" if args.g_reset.lower() == "true" else ""
                    f"--g_penality \n" if args.g_penality.lower() == "true" else ""
                    f"--s_energy_match \n" if args.s_energy_match.lower() == "true" else ""
                    f"--in_energy_target {in_energy_target}\n",
                    f"--out_energy_target {out_energy_target}\n",
                    f"--energy_kd \n" if (args.energy_kd and args.energy_kd.lower() == "true") else "",
                    f"--energy_kd_beta {args.energy_kd_beta}\n" if args.energy_kd_beta else "",
                    "\n",
                    f"--z_dim {z_dim}\n",
                    f"--lr_G {lr_G}\n",
                    f"--lr_S {lr_S}\n",
                    f"--temperature {temperature}\n",
                    f"--oh {oh}\n",
                    f"--act {act}\n",
                    f"--ie {ie}\n",
                    f"--dataset_size {dataset_size}\n",
                    f"--feature_extr \n" if feature_extr == True else "",
                    "\n",
                    f"--train_batch_size {train_batch_size}\n",
                    f"--test_batch_size {test_batch}\n",
                    f"--epochs {epochs}\n",
                    f"--kd_steps {kd_steps}\n",
                    f"--ep_steps {ep_steps}\n",
                    f"--g_steps {g_steps}\n",
                ]
            elif args.method.lower() == "fast":
                lines += [
                    f"--dataset {dataset}\n",
                    f"--dataset_root {dataset_root}\n",
                    f"--num_classes {num_classes}\n",
                    f"--num_workers {num_workers}\n",
                    "\n",
                    f"--save_path {save_path}\n",
                    f"--log_interval {log_interval}\n",
                    "\n",
                    f"--t_network {args.t_network}\n",
                    f"--s_network {args.s_network}\n",
                    "\n",
                    f"--approach {args.approach}\n",
                    f"--method {args.method.lower()}\n",
                    f"--kdci \n" if args.kdci.lower() == "true" else "",
                    f"--confounder_size {confounder_size}\n",
                    f"--hidden_dim {hidden_dim} \n", 
                    f"--ood_loss \n" if args.ood_loss.lower() == "true" else "",
                    f"--additive_loss \n" if (args.additive_loss and args.additive_loss.lower() == "true") else "",
                    f"--gamma_ood {args.gamma_ood}\n" if args.gamma_ood else "",
                    f"--gamma_adaptive {args.gamma_adaptive}\n",
                    f"--new_gamma_ood {args.new_gamma_ood}\n",
                    f"--g_reset \n" if args.g_reset.lower() == "true" else ""
                    f"--g_penality \n" if args.g_penality.lower() == "true" else ""
                    f"--s_energy_match \n" if args.s_energy_match.lower() == "true" else ""
                    f"--in_energy_target {in_energy_target}\n",
                    f"--out_energy_target {out_energy_target}\n",
                    f"--energy_kd \n" if (args.energy_kd and args.energy_kd.lower() == "true") else "",
                    f"--energy_kd_beta {args.energy_kd_beta}\n" if args.energy_kd_beta else "",
                    "\n",
                    f"--z_dim {z_dim}\n",
                    f"--lr_G {lr_G}\n",
                    f"--lr_S {lr_S}\n",
                    f"--temperature {temperature}\n",
                    f"--oh {oh}\n",
                    f"--adv {adv}\n",
                    f"--feat {feat}\n",
                    f"--dataset_size {dataset_size}\n",
                    f"--feature_extr \n" if feature_extr == True else "",
                    "\n",
                    f"--train_batch_size {train_batch_size}\n",
                    f"--test_batch_size {test_batch}\n",
                    f"--epochs {epochs}\n",
                    f"--kd_steps {kd_steps}\n",
                    f"--ep_steps {ep_steps}\n",
                    f"--g_steps {g_steps}\n",
                    f"--warmup {warmup}\n",
                    f"--lr_z {lr_z}\n",
                    f"--data_clear {data_clear}\n",
                    f"--keep_last {keep_last}\n"
                ]
            elif args.method.lower() == "cmi":
                lines += [
                    f"--dataset {dataset}\n",
                    f"--dataset_root {dataset_root}\n",
                    f"--num_classes {num_classes}\n",
                    f"--num_workers {num_workers}\n",
                    "\n",
                    f"--save_path {save_path}\n",
                    f"--log_interval {log_interval}\n",
                    "\n",
                    f"--t_network {args.t_network}\n",
                    f"--s_network {args.s_network}\n",
                    "\n",
                    f"--approach {args.approach}\n",
                    f"--method {args.method.lower()}\n",
                    f"--kdci \n" if args.kdci.lower() == "true" else "",
                    f"--confounder_size {confounder_size}\n",
                    f"--hidden_dim {hidden_dim} \n", 
                    f"--ood_loss \n" if args.ood_loss.lower() == "true" else "",
                    f"--in_energy_target {in_energy_target}\n",
                    f"--out_energy_target {out_energy_target}\n",
                    "\n",
                    f"--z_dim {z_dim}\n",
                    f"--lr_G {lr_G}\n",
                    f"--lr_S {lr_S}\n",
                    f"--temperature {temperature}\n",
                    f"--beta_1 {beta_1}\n",
                    f"--beta_2 {beta_2}\n",
                    f"--dataset_size {dataset_size}\n",
                    f"--feature_extr \n" if feature_extr == True else "",
                    "\n",
                    f"--train_batch_size {train_batch_size}\n",
                    f"--test_batch_size {test_batch}\n",
                    f"--epochs {epochs}\n",
                    f"--kd_steps {kd_steps}\n",
                    f"--ep_steps {ep_steps}\n",
                    f"--g_steps {g_steps}\n",
                    f"--adv {adv}\n",
                    f"--bn {bn}\n",
                    f"--oh {oh}\n",
                    f"--cr {cr}\n",
                    f"--cr_T {cr_T}\n",
                    f"--data_clear {data_clear}\n",
                    f"--keep_last {keep_last}\n"
                ]
            elif args.method.lower() == "deepinv":
                lines += [
                    f"--dataset {dataset}\n",
                    f"--dataset_root {dataset_root}\n",
                    f"--num_classes {num_classes}\n",
                    f"--num_workers {num_workers}\n",
                    "\n",
                    f"--save_path {save_path}\n",
                    f"--log_interval {log_interval}\n",
                    "\n",
                    f"--t_network {args.t_network}\n",
                    f"--s_network {args.s_network}\n",
                    "\n",
                    f"--approach {args.approach}\n",
                    f"--method {args.method.lower()}\n",
                    f"--kdci \n" if args.kdci.lower() == "true" else "",
                    f"--confounder_size {confounder_size}\n",
                    f"--hidden_dim {hidden_dim} \n", 
                    f"--ood_loss \n" if args.ood_loss.lower() == "true" else "",
                    f"--in_energy_target {in_energy_target}\n",
                    f"--out_energy_target {out_energy_target}\n",
                    "\n",
                    f"--lr_G {lr_G}\n",
                    f"--lr_S {lr_S}\n",
                    f"--temperature {temperature}\n",
                    f"--beta_1 {beta_1}\n",
                    f"--beta_2 {beta_2}\n",
                    f"--dataset_size {dataset_size}\n",
                    f"--feature_extr \n" if feature_extr == True else "",
                    "\n",
                    f"--train_batch_size {train_batch_size}\n",
                    f"--test_batch_size {test_batch}\n",
                    f"--epochs {epochs}\n",
                    f"--kd_steps {kd_steps}\n",
                    f"--ep_steps {ep_steps}\n",
                    f"--g_steps {g_steps}\n",
                    f"--adv {adv}\n",
                    f"--bn {bn}\n",
                    f"--oh {oh}\n",
                    f"--tv {tv}\n",
                    f"--data_clear {data_clear}\n",
                    f"--keep_last {keep_last}\n"
                ]
            else:
                print_error("Invalid method.")
                sys.exit(1)
        
        elif args.approach == "sampling":
            if args.method not in SAMPLING_METHODS:
                print_error(f"Method '{args.method}' invalid for sampling approach. Choose one of {SAMPLING_METHODS}.")
                sys.exit(1)
            
            if args.method.lower() == "mosaick":
                lines += [
                    f"--dataset {dataset}\n",
                    f"--ood_dataset {ood_dataset}\n",
                    f"--dataset_root {dataset_root}\n",
                    f"--num_classes {num_classes}\n",
                    f"--ood_classes {ood_classes}\n",
                    f"--num_workers {num_workers}\n",
                    "\n",
                    f"--save_path {save_path}\n",
                    f"--log_interval {log_interval}\n",
                    "\n",
                    f"--t_network {args.t_network}\n",
                    f"--s_network {args.s_network}\n",
                    f"--approach {args.approach}\n",
                    f"--method {args.method.lower()}\n",
                    f"--kdci \n" if args.kdci.lower() == "true" else "",
                    f"--confounder_size {confounder_size}\n",
                    f"--hidden_dim {hidden_dim} \n", 
                    "\n",
                    f"--z_dim {z_dim}\n",
                    f"--lr_S {lr_S}\n",
                    f"--lr_G {lr_G}\n",
                    f"--beta_1 {beta_1}\n",
                    f"--beta_2 {beta_2}\n",
                    f"--kd_steps {kd_steps}\n",
                    f"--feature_extr \n" if feature_extr else "",
                    f"--align {align}\n",
                    f"--local {local}\n",
                    f"--adv {adv}\n",
                    f"--train_batch_size {train_batch_size}\n",
                    f"--test_batch_size {test_batch}\n",
                    f"--epochs {epochs}\n",
                    f"--nc {nc}\n",
                    f"--img_size {img_size}\n",
                    f"--ndf {ndf}"
                ]
            elif args.method.lower() == "dfnd":
                lines += [
                    f"--dataset {dataset}\n",
                    f"--ood_dataset {ood_dataset}\n",
                    f"--dataset_root {dataset_root}\n",
                    f"--num_classes {num_classes}\n",
                    f"--num_workers {num_workers}\n",
                    "\n",
                    f"--save_path {save_path}\n",
                    f"--log_interval {log_interval}\n",
                    "\n",
                    f"--t_network {args.t_network}\n",
                    f"--s_network {args.s_network}\n",
                    f"--approach {args.approach}\n",
                    f"--method {args.method.lower()}\n",
                    f"--kdci \n" if args.kdci.lower() == "true" else "",
                    f"--confounder_size {confounder_size}\n",
                    f"--hidden_dim {hidden_dim} \n", 
                    "\n",
                    f"--num_select {num_select}\n",
                    f"--lr_S {lr_S}\n",
                    f"--lr_N {lr_N}\n",
                    f"--feature_extr \n" if feature_extr else "",
                    f"--train_batch_size {train_batch_size}\n",
                    f"--test_batch_size {test_batch}\n",
                    f"--epochs {epochs}\n",
                ]

            else:
                print_error("Invalid method")
                sys.exit(1)
        
            if mode == "plot":
                lines += [f"--calc_target\n" if args.calc_target.lower() == "true" else "\n"]
        else:
            print_error("Invalid approach")
            sys.exit(1)
        
    elif mode == "test":
        if not args.network:
            print_error("Test mode requires --network.")
            sys.exit(1)
        lines += [
            f"--dataset {dataset}\n",
            f"--dataset_root {dataset_root}\n",
            f"--num_classes {num_classes}\n",
            f"--num_workers {num_workers}\n",
            "\n",
            f"--save_path {save_path}\n",
            "\n",
            f"--network {args.network}\n",
            "\n",
            f"--test_batch_size {test_batch}\n"
            "\n",
            f"--calc_target\n" if args.calc_target.lower() == "true" else "\n"
        ]
    else:
        print_error("Invalid mode")
        sys.exit(1)

    # Ensure output directory exists
    if args.suffix and (mode == "dfkd" or mode == "plot"):
        lines.append(f"--suffix {args.suffix}\n")

    config_dir = os.path.dirname(args.config_file)
    if config_dir and not os.path.exists(config_dir):
        os.makedirs(config_dir, exist_ok=True)
        print_log(f"Created directory: {config_dir}")

    # Write out the config file
    try:
        with open(args.config_file, "w") as f:
            f.writelines(lines)
        print_success(f"Configuration file created: {args.config_file}")
    except Exception as e:
        print_error(f"Error writing '{args.config_file}': {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
