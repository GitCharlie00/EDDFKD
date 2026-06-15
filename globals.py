import numpy as np
import os
import random
import torch

os.environ['MPLCONFIGDIR'] = "/work/project"
os.environ["PYTHONASHSEED"] = "42"
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6,7"
random.seed(42)
np.random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)