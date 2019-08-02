
from IPython.core.interactiveshell import InteractiveShell
import seaborn as sns
# PyTorch
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler
import torch.nn as nn

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Data science tools
import numpy as np
import pandas as pd
import os

# Image manipulations
from PIL import Image
# Useful for examining network
from torchsummary import summary
# Timing utility
from timeit import default_timer as timer

# Visualizations
import matplotlib.pyplot as plt
#%matplotlib inline
plt.rcParams['font.size'] = 14

# Printing out all outputs
InteractiveShell.ast_node_interactivity = 'all'


def setParameters():
    print("Initialize program parameters.")

def eda():
    print("Perform exploratory data analysis.")    

def dataAugmentation():
    print("Image transformation.")
    
def loadPreTrainedModel():
    print("load in the pre-trained model from pytorch.")
    
def adjustClassifiers():
    print("use our own custom classifiers.")
    
def displayResults():
    print("display analysis.")
    
setParameters();
eda();
dataAugmentation();
loadPreTrainedModel();
adjustClassifiers();
displayResults();