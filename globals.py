import numpy as np
import os
import random
import torch

os.environ['MPLCONFIGDIR'] = "/work/project"
os.environ["PYTHONASHSEED"] = "42"
# GPU visibility is controlled by Docker's `--gpus` flag; do NOT hardcode it
# here (the old "3,4,5,6,7" was specific to the original multi-GPU server and
# makes CUDA unavailable on hosts without those device indices).
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6,7"
random.seed(42)
np.random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)