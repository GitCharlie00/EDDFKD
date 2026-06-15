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

def print_log(message: str):
    print(f"{BLUE}[LOG]{RESET} {message}")

def print_error(message: str):
    print(f"{BOLD}{RED}[ERROR]{RESET} {message}")

def print_success(message: str):
    print(f"{BOLD}{GREEN}{message}{RESET}")

def print_warning(message: str):
    print(f"{BOLD}{YELLOW}[WARNING]{RESET} {message}")

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

# Attualmente:W
# - cifar 10: ("resnet-34", "s_resnet-18"),
# - cifar 10: ("vgg-11", "s_resnet-18"),
# - cifar 10: ("wrn-40-2", "s_wrn-16-1"),
# - cifar 10: ("wrn-40-2", "s_wrn-40-1"),
# - cifar 10: ("wrn-40-2", "s_wrn-16-2")
# - cifar 100: ("resnet-34", "s_resnet-18"),
# - cifar 100: ("vgg-11", "s_resnet-18"),
# - cifar 100: ("wrn-40-2", "s_wrn-16-1"),
# - cifar 100: ("wrn-40-2", "s_wrn-40-1"),
# - cifar 100: ("wrn-40-2", "s_wrn-16-2")

def get_all_datasets() -> list[str]:
    return ["cifar10"] #["cifar10", "cifar100"]

def get_distillation_pairs() -> list[tuple[str, str]]:
    """Returns all teacher/student pairs to test"""
    return [
        ("resnet-34", "s_resnet-18"),
        # ("vgg-11", "s_resnet-18"),
        # ("wrn-40-2", "s_wrn-16-1"),
        # ("wrn-40-2", "s_wrn-40-1"),
        # ("wrn-40-2", "s_wrn-16-2")
    ]

def prompt_gpu_id() -> int:
    """Prompt user for GPU ID to use"""
    while True:
        gpu_input = input(f"{BOLD}{YELLOW}Enter GPU ID to use (0, 1, 2, 3, etc.) or press Enter for default: {RESET}").strip()
        if not gpu_input:  # Empty input, use default
            return None
        if gpu_input.isdigit():
            return int(gpu_input)
        print_error("Please enter a valid GPU ID number.")

def prompt_generative_method() -> str:
    return prompt_menu("Select generative DFKD method:", ["DAFL", "Fast", "CMI", "DeepInv"])

def run_create_arguments_file(project_dir, main_script, config_file, args: list[str]) -> bool:
    cmd = [sys.executable, os.path.join(project_dir, "create_arguments_file.py"), main_script, config_file] + args
    print_log(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print_error("create_arguments_file.py failed.")
        return False
    print_success("Arguments file created successfully.")
    return True

def run_docker(project_dir, data_dir, main_script, config_file, gpu_id=None) -> bool:
    image = "claudioschi21/thesis_alcor_cuda11.8:latest"
    cmd = [
        "podman", "run",
        "-v", f"{project_dir}:/work/project",
        "-v", f"{data_dir}:/work/data",
        "--device", "nvidia.com/gpu=all",
        "--ipc", "host"
    ]
    
    cmd.extend([
        image,
        "/usr/bin/python3",
        main_script,
        config_file
    ])
    
    print_log(f"Running Podman: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print_error("Podman execution failed.")
        return False
    print_success("Podman completed successfully.")
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

def run_dfkd_experiment(project_dir, data_dir, configs_dir, main_script, dataset, teacher, student, method, kdci, ood_loss, experiment_name, gpu_id=None):
    """Run a single DFKD experiment"""
    print(f"\n{BOLD}{BLUE}=== Running {experiment_name} ==={RESET}")
    
    args = [
        "--mode", "dfkd",
        "--dataset", dataset,
        "--t_network", teacher,
        "--s_network", student,
        "--approach", "generative",
        "--method", method,
        "--kdci", "true" if kdci else "false",
        "--ood_loss", "true" if ood_loss else "false",
        "--gpu_id", gpu_id
    ]
    
    # Create config filename
    kdci_suffix = "_kdci" if kdci else ""
    ood_suffix = "_ood" if ood_loss else ""
    config_base = f"dfkd_{method.lower()}_{dataset}_{teacher.replace('-', '_')}_{student.replace('-', '_')}{kdci_suffix}{ood_suffix}.txt"
    host_cfg = os.path.join(configs_dir, config_base)
    cont_cfg = os.path.join("/work/project", "configs", config_base)
    
    # Create arguments file
    if not run_create_arguments_file(project_dir, main_script, host_cfg, args):
        print_error(f"Failed to create config for {experiment_name}")
        return False
    
    # Run experiment
    if not run_docker(project_dir, data_dir, main_script, cont_cfg, gpu_id):
        print_error(f"Failed to run {experiment_name}")
        return False
    
    print_success(f"Completed {experiment_name}")
    return True

def run_plot_experiment(project_dir, data_dir, configs_dir, main_script, dataset, teacher, student, method, kdci, ood_loss, plot_name, gpu_id=None):
    """Run plot experiment for the method with specific KDCI/OOD configuration"""
    print(f"\n{BOLD}{BLUE}=== Generating {plot_name} ==={RESET}")
    
    args = [
        "--mode", "plot",
        "--dataset", dataset,
        "--t_network", teacher,
        "--s_network", student,
        "--approach", "generative",
        "--method", method,
        "--kdci", "true" if kdci else "false",
        "--ood_loss", "true" if ood_loss else "false",
        "--gpu_id", gpu_id
    ]
    
    print_log(f"Plot args: {args}")  # Debug line to verify parameters
    
    # Create config filename that matches the DFKD experiment
    kdci_suffix = "_kdci" if kdci else ""
    ood_suffix = "_ood" if ood_loss else ""
    config_base = f"plot_{method.lower()}_{dataset}_{teacher.replace('-', '_')}_{student.replace('-', '_')}{kdci_suffix}{ood_suffix}.txt"
    host_cfg = os.path.join(configs_dir, config_base)
    cont_cfg = os.path.join("/work/project", "configs", config_base)
    
    # Create arguments file
    if not run_create_arguments_file(project_dir, main_script, host_cfg, args):
        print_error(f"Failed to create plot config for {plot_name}")
        return False
    
    # Run plot
    if not run_docker(project_dir, data_dir, main_script, cont_cfg, gpu_id):
        print_error(f"Failed to generate {plot_name}")
        return False
    
    print_success(f"Completed {plot_name}")
    return True

def run_method_complete_pipeline(project_dir, data_dir, configs_dir, main_script, method, gpu_id):
    """Run complete pipeline for a method across all datasets and distillation pairs"""
    print(f"\n{BOLD}{GREEN}=== Starting complete pipeline for {method} ==={RESET}")
    
    datasets = get_all_datasets()
    distillation_pairs = get_distillation_pairs()
    
    total_combinations = len(datasets) * len(distillation_pairs)
    print_log(f"Testing {len(datasets)} datasets × {len(distillation_pairs)} distillation pairs = {total_combinations} combinations")
    
    successful_combinations = []
    failed_combinations = []
    
    for dataset in datasets:
        for teacher, student in distillation_pairs:
            combination_name = f"{dataset} - {teacher}/{student}"
            print(f"\n{BOLD}{YELLOW}=== Testing {combination_name} ==={RESET}")
            
            result = run_single_combination_pipeline(project_dir, data_dir, configs_dir, main_script, 
                                                   dataset, teacher, student, method, combination_name, gpu_id)
            
            if result['success']:
                successful_combinations.append({
                    'name': combination_name,
                    'dfkd_success': result['dfkd_success'],
                    'plots_success': result['plots_success'],
                    'failed_experiments': result['failed_experiments']
                })
                print_success(f"✓ Completed {combination_name}")
            else:
                failed_combinations.append({
                    'name': combination_name,
                    'dfkd_success': result['dfkd_success'],
                    'plots_success': result['plots_success'],
                    'failed_experiments': result['failed_experiments']
                })
                print_error(f"✗ Failed {combination_name}")
    
    # Final summary
    print(f"\n{BOLD}{GREEN}=== {method.upper()} PIPELINE SUMMARY ==={RESET}")
    print_log(f"Total combinations: {total_combinations}")
    print_success(f"Successful combinations: {len(successful_combinations)}")
    
    if failed_combinations:
        print_error(f"Failed combinations: {len(failed_combinations)}")
        print(f"\n{BOLD}{RED}=== DETAILED FAILURE REPORT ==={RESET}")
        
        for failed in failed_combinations:
            print(f"\n{BOLD}{RED}❌ {failed['name']}{RESET}")
            print(f"   DFKD experiments: {failed['dfkd_success']}/3")
            print(f"   Plot generations: {failed['plots_success']}/3")
            
            if failed['failed_experiments']:
                print(f"   {BOLD}Failed experiments:{RESET}")
                for exp in failed['failed_experiments']:
                    print(f"     • {exp}")
    
    # Show successful combinations with partial failures
    partial_success = [combo for combo in successful_combinations if combo['failed_experiments']]
    if partial_success:
        print(f"\n{BOLD}{YELLOW}=== COMBINATIONS WITH PARTIAL FAILURES ==={RESET}")
        for combo in partial_success:
            print(f"\n{BOLD}{YELLOW}⚠️  {combo['name']}{RESET}")
            print(f"   DFKD experiments: {combo['dfkd_success']}/3")
            print(f"   Plot generations: {combo['plots_success']}/3")
            print(f"   {BOLD}Failed experiments:{RESET}")
            for exp in combo['failed_experiments']:
                print(f"     • {exp}")
    
    if not failed_combinations:
        print_success("All combinations completed successfully!")
    
    return len(successful_combinations) > 0

def run_single_combination_pipeline(project_dir, data_dir, configs_dir, main_script, dataset, teacher, student, method, combination_name, gpu_id):
    """Run complete pipeline for a single dataset/teacher/student combination"""
    print_log(f"Starting pipeline for {combination_name}")
    
    experiments = [
        (False, False, f"{method} (vanilla)"),
        (True, False, f"{method} + KDCI"),
        (False, True, f"{method} + OOD"),
    ]
    
    successful_dfkd = 0
    successful_plots = 0
    failed_experiments = []
    
    # Run DFKD experiments and their corresponding plots
    for kdci, ood_loss, exp_name in experiments:
        exp_full_name = f"{combination_name} - {exp_name}"
        dfkd_success = False
        plot_success = False
        
        # Run DFKD experiment
        if run_dfkd_experiment(project_dir, data_dir, configs_dir, main_script, 
                              dataset, teacher, student, method, kdci, ood_loss, exp_full_name, gpu_id):
            successful_dfkd += 1
            dfkd_success = True
            
            # Run corresponding plot
            plot_name = f"{combination_name} - {exp_name} Plot"
            if run_plot_experiment(project_dir, data_dir, configs_dir, main_script, 
                                  dataset, teacher, student, method, kdci, ood_loss, plot_name, gpu_id):
                successful_plots += 1
                plot_success = True
            else:
                failed_experiments.append(f"{exp_name} Plot")
                print_warning(f"DFKD experiment successful but plot generation failed for {exp_full_name}")
        else:
            failed_experiments.append(f"{exp_name} DFKD")
            failed_experiments.append(f"{exp_name} Plot")
            print_warning(f"Experiment {exp_full_name} failed, skipping its plot...")
    
    # Summary for this combination
    print_log(f"Pipeline completed for {combination_name}:")
    print_log(f"  - DFKD experiments: {successful_dfkd}/3")
    print_log(f"  - Plot generations: {successful_plots}/3")
    
    # Consider successful if at least one complete DFKD experiment succeeded
    success = successful_dfkd > 0
    
    return {
        'success': success,
        'dfkd_success': successful_dfkd,
        'plots_success': successful_plots,
        'failed_experiments': failed_experiments
    }

def main():
    project_dir = os.path.abspath(os.getcwd())
    data_dir    = os.path.join("/mnt/ssd1/schiavella/", "data")
    configs_dir = os.path.join(project_dir, "configs")
    main_script = "/work/project/main.py"
    
    print(f"{BOLD}{GREEN}=== DFKD Generative Methods Complete Pipeline ==={RESET}")
    print("This script will run a complete pipeline for the selected generative DFKD method.")
    print("It will test ALL combinations of:")
    print("  • Datasets: cifar10, cifar100")
    print("  • Distillation pairs:")
    print("    - resnet-34 → s_resnet-18")
    print("    - vgg-11 → s_resnet-18") 
    print("    - wrn-40-2 → s_wrn-16-1")
    print("    - wrn-40-2 → s_wrn-40-1")
    print("    - wrn-40-2 → s_wrn-16-2")
    print("\nFor each combination, it will run:")
    print("  1. DFKD (vanilla) + Plot")
    print("  2. DFKD + KDCI + Plot") 
    print("  3. DFKD + OOD + Plot")
    
    # Get method and GPU selection
    method = prompt_generative_method()
    gpu_id = prompt_gpu_id()
    
    datasets = get_all_datasets()
    distillation_pairs = get_distillation_pairs()
    total_combinations = len(datasets) * len(distillation_pairs)
    total_experiments = total_combinations * 6  # 3 DFKD + 3 plots per combination
    
    print(f"\n{BOLD}{BLUE}=== Pipeline Summary ==={RESET}")
    print_log(f"Selected method: {method}")
    if gpu_id is not None:
        print_log(f"Selected GPU: {gpu_id}")
    else:
        print_log("Using default GPU configuration")
    print_log(f"Total combinations to test: {total_combinations}")
    print_log(f"Total experiments to run: {total_experiments} (3 DFKD + 3 plots per combination)")
    print_log(f"Estimated duration: This will take several hours!")
    
    if not prompt_yes_no("Do you want to proceed with the complete pipeline?"):
        print_success("Canceled.")
        sys.exit(0)
    
    # Clean configs directory
    # clean_configs_dir(configs_dir)
    
    # Run complete pipeline for the selected method
    success = run_method_complete_pipeline(project_dir, data_dir, configs_dir, main_script, method, str(gpu_id))
    
    if success:
        print(f"\n{BOLD}{GREEN}=== COMPLETE PIPELINE FINISHED SUCCESSFULLY ==={RESET}")
        print_success(f"All experiments for {method} across all combinations have been completed!")
    else:
        print(f"\n{BOLD}{RED}=== PIPELINE COMPLETED WITH ERRORS ==={RESET}")
        print_error(f"Some experiments for {method} failed. Check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()