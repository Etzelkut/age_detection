#can be deleted, just made for myself to make evrything clean
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import random
import torch.nn.functional as F
import os
import pandas as pd
from typing import List, Dict, Tuple
from torch import Tensor
import math, copy, time
from torch.autograd import Variable
from einops import rearrange

import torchvision
import torchvision.transforms as transforms
from torchvision import utils
from PIL import Image




device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#seeding can be used when training
def seed_e(seed_value):
  pl.seed_everything(seed_value)
  random.seed(seed_value)
  np.random.seed(seed_value) 
  torch.manual_seed(seed_value)
  torch.cuda.manual_seed(seed_value)
  torch.cuda.manual_seed_all(seed_value)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
