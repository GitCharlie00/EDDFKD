import random
from globals import *
import torch

import torch.backends.cudnn as cudnn

from datetime import datetime
from generative.dfkd_gen import dfkd_gen
from plot import plot
from sampling.dfkd_sam import dfkd_sam
from test import test
from train import train
from utils import *

def main():  
    # Argument parser
    args = init_arg_parser()
    
    formatter = TextFormatter()
    intro_str = formatter.format(
        f"D-DFKD experiments",
        color = 'blue', 
        style = ['bold'], 
        separator = True
    )
    print(intro_str)
    print_args(args)

    # Device setup
    device_str = formatter.format(
        "Setting device", 
        color = 'blue', 
        style = ['bold'], 
        separator = True
    )
    print(device_str)
    device = hardware_check(args.mode,args.gpu_id)
    args.gpu = device
    
    # Starting train
    if args.mode == "train":
        train_str = "Starting training mode"
        train_str = formatter.format(
            train_str, 
            color = 'yellow', 
            style = ['bold'], 
            separator = True
        )
        print(train_str)

        try:
            train(args)
        except KeyboardInterrupt:
            stop_str = formatter.format(
                "Training interrupted by user", 
                color = 'yellow', 
                style = ['underlined'], 
                separator = True
            )
            print(stop_str)
            torch.cuda.empty_cache()
        except Exception as e:
            stop_str = formatter.format(
                "[!!! ERROR !!!]", 
                color = 'red', 
                style = ['bold'], 
                separator = True
            )
            print(stop_str)

            send_email(args.t_network,args.dataset, error=True, error_msg=e)
            torch.cuda.empty_cache()
            raise
            
    # Starting test
    elif args.mode == "test":
        test_str = "Starting testing mode"
        test_str = formatter.format(
            test_str, 
            color = 'yellow', 
            style = ['bold'], 
            separator = True
        )
        print(test_str)

        try:
            test(args)
        except KeyboardInterrupt:
            print("Test interrupted by user")
            torch.cuda.empty_cache()
        except Exception as e:
            stop_str = formatter.format(
                "[!!! ERROR !!!]", 
                color = 'red', 
                style = ['bold'], 
                separator = True
            )
            print(stop_str)
            torch.cuda.empty_cache()
            raise
    elif args.mode == "dfkd":
        dfkd_str = "Starting dfkd mode"
        dfkd_str = formatter.format(
            dfkd_str, 
            color = 'yellow', 
            style = ['bold'], 
            separator = True
        )
        print(dfkd_str)
        
        try:
            if args.approach == "generative":
                if args.method.lower() not in ["dafl", "fast", "cmi", "deepinv"]:
                    raise ValueError("Invalid method specified for generative DFKD. Choose 'dafl', 'fast', 'cmi', or 'deepinv'.")
                dfkd_gen(args)
            elif args.approach == "sampling": 
                if args.method.lower() not in ["mosaick", "dfnd"]:
                    raise ValueError("Invalid method specified for sampling DFKD. Choose 'mosaick', 'dfnd'.")
                dfkd_sam(args)
            else:
                raise ValueError("Invalid approach specified. Choose 'generative' or 'sampling'.")
        except KeyboardInterrupt:
            print("DFKD interrupted by user")
            torch.cuda.empty_cache()
        except Exception as e:
            stop_str = formatter.format(
                "[!!! ERROR !!!]", 
                color = 'red', 
                style = ['bold'], 
                separator = True
            )
            print(stop_str)
            print(e)
            torch.cuda.empty_cache()
            raise
    elif args.mode == "plot":
        plot_str = "Starting plot mode"
        plot_str = formatter.format(
            plot_str, 
            color = 'yellow', 
            style = ['bold'], 
            separator = True
        )
        print(plot_str)

        try:
            approach_suffix = approach_label(args)
            user_suffix = f"_{args.suffix}" if args.suffix else ""
            plot(args,approach_suffix,user_suffix)
        except KeyboardInterrupt:
            print("Plot interrupted by user")
            torch.cuda.empty_cache()
        except Exception as e:
            stop_str = formatter.format(
                "[!!! ERROR !!!]", 
                color = 'red', 
                style = ['bold'], 
                separator = True
            )
            print(stop_str)
            torch.cuda.empty_cache()
            raise
    else:
        mode_str = formatter.format(
            "[!!!ERROR!!!] Choose between train, test, dfkd, and plot. Adjust the --mode flag.",  
            color = 'red', 
            style = ['bold'], 
            separator = False
        )
        print(mode_str)

if __name__ == '__main__':
    main()