import json
import os
import subprocess
import sys

# ANSI escape codes for colors and styles
RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
BLUE   = "\033[34m"

def load_last_config(path):
    if os.path.isfile(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return None

def save_last_config(path, cfg_dict):
    with open(path, 'w') as f:
        json.dump(cfg_dict, f, indent=2)

def print_log(message: str):
    print(f"{BLUE}[LOG]{RESET} {message}")

def print_error(message: str):
    print(f"{BOLD}{RED}[ERROR]{RESET} {message}")

def print_success(message: str):
    print(f"{BOLD}{GREEN}{message}{RESET}")

def print_args_per_line(args: list[str]):
    # Header
    print(f"{BLUE}[LOG]{RESET} Args:")
    # Cicla a coppie (flag, valore)
    for flag, val in zip(args[::2], args[1::2]):
        print(f"       {BLUE}{flag}{RESET} {val}")
    # Se c’è un flag orfano (senza valore), lo stampiamo comunque
    if len(args) % 2 == 1:
        orphan = args[-1]
        print(f"    {BLUE}{orphan}{RESET}")

def prompt_menu(message: str, options: list[str]) -> str:
    print(f"\n{BOLD}{YELLOW}{message}{RESET}")
    for idx, opt in enumerate(options, start=1):
        print(f"  {BOLD}[{idx}]{RESET} {opt}")
    while True:
        choice = input(f"{BOLD}Select option [1-{len(options)}]: {RESET}")
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print_error("Invalid selection. Try again.")

def prompt_yes_no(message: str) -> bool:
    response = input(f"{BOLD}{YELLOW}{message} [y/N]: {RESET}").strip().lower()
    return response in ("y", "yes")

def prompt_task() -> str:
    return prompt_menu("Select task type:", ["train", "test", "dfkd", "plot"])

def prompt_suffix() -> str:
    if prompt_yes_no("Do you want to add a custom suffix to the folder name?"):
        suffix = input(f"{BOLD}Enter suffix (no spaces): {RESET}").strip()
        return suffix
    return ""

def prompt_gpu_id():
    """Prompt user for GPU ID to use"""
    while True:
        gpu_input = input(f"{BOLD}{YELLOW}Enter GPU ID to use (0, 1, 2, 3, etc.) or press Enter for default: {RESET}").strip()
        if not gpu_input:  # Empty input, use default
            return None
        if gpu_input.isdigit():
            return gpu_input
        print_error("Please enter a valid GPU ID number.")

def prompt_float(param_name: str, default_val: str) -> str:
    """Richiede un numero decimale all'utente, con opzione di default."""
    while True:
        val_input = input(f"{BOLD}{YELLOW}Enter value for {param_name} (default: {default_val}): {RESET}").strip()
        if not val_input:
            return default_val
        try:
            float(val_input) # Controlla se è un numero valido
            return val_input
        except ValueError:
            print_error(f"Please enter a valid number for {param_name}.")

def prompt_dataset() -> str:
    return prompt_menu("Select dataset:", ["cifar10", "cifar100"])

def prompt_architecture(role: str = "") -> str:
    if role:
        msg = f"Select {role} architecture:"
        if role.lower() == "teacher":
            return prompt_menu(msg, ["resnet-34", "vgg-11", "wrn-40-2"])
        elif role.lower() == "student":
            return prompt_menu(msg, ["s_resnet-18", "s_wrn-16-1", "s_wrn-16-2", "s_wrn-40-1"])
        else:
            print_error("Invalid role specified. Use 'teacher' or 'student'.")
            sys.exit(1)
    else:
        return prompt_menu(
            "Select network architecture:",
            ["resnet-34", "vgg-11", "wrn-40-2", "s_resnet-18", "s_wrn-16-1", "s_wrn-16-2", "s_wrn-40-1"]
        )

def prompt_approach() -> str:
    return prompt_menu("Select DFKD approach:", ["generative", "sampling"])

def prompt_method_for(approach: str) -> str:
    if approach == "generative":
        return prompt_menu("Select generative method:", ["DAFL", "Fast", "CMI", "DeepInv"])
    else:  # sampling
        return prompt_menu("Select sampling method:", ["Mosaick", "DFND"])

def prompt_additive_loss() -> bool:
    return prompt_yes_no("Should the energy loss term be additive?")

def prompt_gamma_ood() -> str:
    return prompt_menu("Select OOD gamma value:", ["0.1", "0.3", "0.5", "0.7", "1.0"])

def prompt_adaptive_gamma() -> str:
    return prompt_menu("Select adaptive gamma strategy:", ["single", "double", "none"])

def prompt_new_gamma_ood() -> str:
    return prompt_menu("Select new OOD gamma value for adaptive strategy:", ["0.1", "0.3", "0.5", "0.7", "1.0"])

def prompt_g_reset() -> bool:
    return prompt_yes_no("Should the generator be resetted?")

def prompt_g_penality() -> bool:
    return prompt_yes_no("Should the generator be penalized on similarity?")

def prompt_s_energy_match() -> bool:
    return prompt_yes_no("Should the S and T matching energy?")

def get_all_models() -> list[str]:
    return ["resnet-34", "vgg-11", "wrn-40-2", "s_resnet-18", "s_wrn-16-1", "s_wrn-16-2", "s_wrn-40-1"]

def get_all_datasets() -> list[str]:
    return ["cifar10", "cifar100"]

def run_create_arguments_file(project_dir, main_script, config_file, args: list[str]) -> bool:
    cmd = [sys.executable, os.path.join(project_dir, "create_arguments_file.py"), main_script, config_file] + args
    print("--------------")
    print_log(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print_error("create_arguments_file.py failed.")
        return False
    print_success("Arguments file created successfully.")
    return True

def run_docker(project_dir, data_dir, main_script, config_file) -> bool:
    image = "claudioschi21/thesis_alcor_cuda11.8:latest"
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{project_dir}:/work/project",
        "-v", f"{data_dir}:/work/data",
        "--gpus", "all",
        "--ipc", "host",
        image,
        "/usr/bin/python3", "-u",
        main_script,
        config_file
    ]
    print("--------------")
    print_log(f"Running Docker: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print_error("Docker execution failed.")
        return False
    print_success("Docker completed successfully.")
    return True

def clean_configs_dir(configs_dir):
    if os.path.exists(configs_dir):
        for file in os.listdir(configs_dir):
            path = os.path.join(configs_dir, file)
            try:
                if os.path.isfile(path):
                    os.remove(path)
                    print_log(f"Removed old config: {path}")
            except Exception as e:
                print_error(f"Failed to remove {path}: {e}")
    else:
        os.makedirs(configs_dir)
        print_log(f"Created configs directory: {configs_dir}")

def run_single_test(project_dir, data_dir, configs_dir, main_script, dataset, model):
    print_log(f"Running test for {model} on {dataset}")
    args = ["--mode", "test", "--dataset", dataset, "--network", model]
    basename = f"test_{dataset}_{model.replace('-', '_')}.txt"
    host_cfg = os.path.join(configs_dir, basename)
    cont_cfg = os.path.join("/work/project", "configs", basename)
    if not run_create_arguments_file(project_dir, main_script, host_cfg, args):
        print_error(f"Failed to create config for {model} on {dataset}")
        return False
    if not run_docker(project_dir, data_dir, main_script, cont_cfg):
        print_error(f"Failed to run test for {model} on {dataset}")
        return False
    print_success(f"Completed test for {model} on {dataset}")
    return True

def run_full_test(project_dir, data_dir, configs_dir, main_script):
    models = get_all_models()
    datasets = get_all_datasets()
    total = len(models) * len(datasets)
    print_log(f"Starting full test: {len(models)} models × {len(datasets)} datasets = {total} combinations")
    succ = fail = 0
    for i, ds in enumerate(datasets, 1):
        for j, m in enumerate(models, 1):
            combo = (i - 1) * len(models) + j
            print(f"\n{BOLD}{BLUE}=== Combination {combo}/{total}: {m} on {ds} ==={RESET}")
            if run_single_test(project_dir, data_dir, configs_dir, main_script, ds, m):
                succ += 1
            else:
                fail += 1
                print_error(f"Test failed for {m} on {ds}")
    print(f"\n{BOLD}{GREEN}=== FULL TEST SUMMARY ==={RESET}")
    print_log(f"Total combinations: {total}")
    print_success(f"Successful runs: {succ}")
    if fail:
        print_error(f"Failed runs: {fail}")
    else:
        print_success("All tests completed successfully!")

def main():
    project_dir = os.path.abspath(os.getcwd())
    data_dir    = os.path.join("/mnt/ssd1/schiavella/", "data")
    configs_dir = os.path.join(project_dir, "configs")
    main_script = "/work/project/main.py"

    last_cfg_file = "./last_config.json"
    last = load_last_config(last_cfg_file)
    skip = False

    if last:
        print_log("Previous configuration found:")
        print_args_per_line(last['args'])
        if prompt_yes_no("Would you like to relaunch this configuration?"):
            skip = True
        else:
            skip = False

    if not skip:
        task = prompt_task()

        # full-test for "test" mode
        if task == "test":
            if prompt_yes_no("Test all models on all datasets?"):
                print_log("Full test mode selected")
                clean_configs_dir(configs_dir)
                if not prompt_yes_no("Proceed with full test?"):
                    print_success("Canceled.")
                    sys.exit(0)
                run_full_test(project_dir, data_dir, configs_dir, main_script)
                sys.exit(0)
            else:
                print_log("Single test mode")

        # build args for train/test/dfkd
        args = ["--mode", task]

        if task in ("train", "test"):
            ds   = prompt_dataset()
            arch = prompt_architecture()
            gpu_id = prompt_gpu_id()
            args += ["--dataset", ds, "--network", arch, "--gpu_id", str(gpu_id) if gpu_id is not None else "0"]
            config_file_name = f"{task}_{ds}_{arch}.txt"

        else:  # dfkd and plot
            ds        = prompt_dataset()
            t_arch    = prompt_architecture("teacher")
            s_arch    = prompt_architecture("student")
            approach  = prompt_approach()
            method    = prompt_method_for(approach)
            use_kdci  = prompt_yes_no("Do you want to integrate KDCI?")
            use_ood   = prompt_yes_no("Do you want to integrate OOD loss?")
            use_energy_kd = prompt_yes_no("Do you want energy-weighted KD?")
            use_energy_temp = prompt_yes_no("Do you want energy-adaptive KD temperature?")
            if task == "plot":
                calc_target = prompt_yes_no("Do you want to compute energy target?")

            additive_loss = False
            gamma_ood = "0.0"
            gamma_adaptive = "none"
            new_gamma_ood = "0.0"
            g_reset = False
            g_penality = False
            s_energy_match = False
            # gamma_adaptive = prompt_adaptive_gamma()
            # new_gamma_ood = prompt_new_gamma_ood() if gamma_adaptive != "none" else "0.0"
            # g_reset = prompt_g_reset()
            # g_penality = prompt_g_penality()
            # s_energy_match = prompt_s_energy_match()

            args += [
                "--dataset", ds,
                "--t_network", t_arch,
                "--s_network", s_arch,
                "--approach", approach,
                "--method", method,
                "--kdci", "true" if use_kdci else "false",
                "--ood_loss", "true" if use_ood else "false",
                "--energy_kd", "true" if use_energy_kd else "false",
                "--energy_temp", "true" if use_energy_temp else "false",
                "--gamma_adaptive", gamma_adaptive,
                "--new_gamma_ood", new_gamma_ood,
                "--g_reset", "true" if g_reset else "false",
                "--g_penality", "true" if g_penality else "false",
                "--s_energy_match", "true" if s_energy_match else "false",
            ]

            if method == "DAFL":
                if use_ood:
                    # gamma_ood = "0.2" #prompt_gamma_ood()
                    gamma_ood = prompt_float("gamma_ood", "0.05")
                    val_lr_G  = prompt_float("lr_G", "0.002") 
                    val_oh    = prompt_float("oh", "0.5")
                    val_act   = prompt_float("act", "0.1")
                    val_ie    = prompt_float("ie", "20.0")
                    args += [
                        "--lr_G", str(val_lr_G),
                        "--oh", str(val_oh),
                        "--act", str(val_act),
                        "--ie", str(val_ie)
                    ]

            elif method == "Fast":
                if use_ood:
                    # gamma_ood = "0.2" #prompt_gamma_ood()
                    gamma_ood = prompt_float("gamma_ood", "0.05")
                    val_lr_G  = prompt_float("lr_G", "0.002") 
                    val_oh    = prompt_float("oh", "0.4")
                    val_adv   = prompt_float("adv", "1.1")
                    val_feat  = prompt_float("feat", "10.0")

                    args += [
                        "--lr_G", str(val_lr_G),
                        "--oh", str(val_oh),
                        "--adv", str(val_adv),
                        "--feat", str(val_feat)
                    ]


            if use_energy_temp:
                # base tau_0 (try 4/8/20) and adaptivity strength (0 = global T baseline D0, 1 = energy-adaptive D)
                val_temp_base  = prompt_float("energy_temp_base", "8.0")
                val_temp_alpha = prompt_float("energy_temp_alpha (0=global T)", "1.0")
                args += [
                    "--energy_temp_base", str(val_temp_base),
                    "--energy_temp_alpha", str(val_temp_alpha)
                ]

            gpu_id = prompt_gpu_id()
            additive_loss = True #prompt_additive_loss()

            if use_ood and additive_loss is not None:
                args += ["--additive_loss", "true" if additive_loss else "false"]
            if use_ood and gamma_ood is not None:
                args += ["--gamma_ood", gamma_ood]

            user_suffix = prompt_suffix()
            if user_suffix:
                args += ["--suffix", user_suffix]

            if task == "plot":
                args += ["--calc_target", "true" if calc_target else "false"]

            args += ["--gpu_id", str(gpu_id) if gpu_id is not None else "0"]

            parts = []
            if use_ood:
                prefix = "add_" if additive_loss else ""
                
                if gamma_adaptive != "none":
                    old_g = str(gamma_ood)       # Il vecchio gamma
                    new_g = str(new_gamma_ood)   # Il nuovo gamma
                    ood_part = f"{prefix}ood_{gamma_adaptive}_{old_g}_to_{new_g}"
                else:
                    ood_part = f"{prefix}ood_{str(gamma_ood)}"
                parts.append(ood_part)
            if use_kdci:
                parts.append("kdci")
            if use_energy_kd:
                parts.append("ekd")
            if use_energy_temp:
                parts.append("etemp")
            if g_reset:
                parts.append("g_reset")
            if g_penality:
                parts.append("g_penality")
            if s_energy_match:
                parts.append("match")
            if user_suffix:
                parts.append(user_suffix)
                
            # Unisci tutte le parti e aggiungi l'underscore iniziale
            suffix = f"_{'_'.join(parts)}"

            config_file_name = f"{task}_{ds}_{t_arch}_{s_arch}_{approach}_{method}{suffix}.txt"
            

    cfg_base  = config_file_name if not skip else last['task']
    host_cfg  = os.path.join(configs_dir, cfg_base) if not skip else last['host_cfg']
    cont_cfg  = os.path.join("/work/project", "configs", cfg_base) if not skip else last['cont_cfg']
    args = args if not skip else last['args']
    task = task if not skip else last['task']

    print_log(f"Selected task: {task}")
    print_log(f"Config will be: {cont_cfg}")

    # clean_configs_dir(configs_dir)
    save_last_config(last_cfg_file, {
        "task":     task,
        "args":     args,
        "host_cfg": host_cfg,
        "cont_cfg": cont_cfg
    })

    if not run_create_arguments_file(project_dir, main_script, host_cfg, args):
        sys.exit(1)

    if not prompt_yes_no("Proceed with Docker?"):
        print_success("Canceled.")
        sys.exit(0)

    if not run_docker(project_dir, data_dir, main_script, cont_cfg):
        sys.exit(1)

if __name__ == "__main__":
    main()

