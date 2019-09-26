# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 07:33:14 2019

@author: vivek
"""

# Import required libraries

from torchvision import transforms, datasets, models
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler
import torch.nn as nn

import numpy as np
import pandas as pd

import os

from PIL import Image
from torchsummary import summary
from timeit import default_timer as timer

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

trained_model = False
batch_size = 16

train_on_gpu = cuda.is_available()
print(f'Train on gpu: {train_on_gpu}')

datadir = 'Data/'
traindir = datadir + 'train/'
validdir = datadir + 'valid/'
testdir = datadir + 'test/'
categories = []
img_categories = []
n_train = []
n_valid = []
n_test = []
hs = []
ws = []

for d in os.listdir(traindir):
    categories.append(d)
    train_imgs = os.listdir(traindir + d)
    valid_imgs = os.listdir(validdir + d)
    test_imgs = os.listdir(testdir + d)
    n_train.append(len(train_imgs))
    n_valid.append(len(valid_imgs))
    n_test.append(len(test_imgs))
    
    for i in train_imgs:
        img_categories.append(d)
        img = Image.open(traindir + d + '/' + i)
        img_array = np.array(img)
        hs.append(img_array.shape[0])
        ws.append(img_array.shape[1])
    
cat_df = pd.DataFrame({'category': categories, 
                         'n_train': n_train,
                         'n_valid': n_valid, 'n_test': n_test}).\
              sort_values('category')

image_df = pd.DataFrame({'category': img_categories, 
                        'height': hs, 'width': ws})

cat_df.tail()

cat_df.to_csv('categories.csv', index = False)

cat_df.set_index('category')['n_train'].plot.bar(color = 'r',
                                             figsize = (20, 6))
plt.xticks(rotation = 80);

image_df.groupby('category').describe()

def imshow(image):
    plt.figure(figsize = (6, 6))
    plt.imshow(image)
    plt.axis('off');
    plt.show();
    
x = Image.open(datadir + 'ewer/image_0002.jpg')
np.array(x).shape

imshow(x);

image_transforms = {
        'train': transforms.Compose([
                 transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                 transforms.RandomHorizontalFlip(),
                 transforms.CenterCrop(size=224),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
     'test': transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

batch_size = 16

# Datasets
data = {'train': datasets.ImageFolder(root=traindir,
                                      transform=image_transforms['train']),
        'val': datasets.ImageFolder(root=validdir,
                                     transform=image_transforms['val']),
        'test': datasets.ImageFolder(root=testdir,
                                     transform=image_transforms['test'])
       }

dataloaders = {'train': DataLoader(data['train'], batch_size=batch_size),
              'val': DataLoader(data['val'], batch_size=batch_size),
               'test': DataLoader(data['test'], batch_size=batch_size)
              }


%%capture
for i, row in cat_df.iterrows():
    train_num = int(0.75 * row['n'])
    images = os.listdir(datadir + row['category'])
    train_imgs = list(np.random.choice(images, 
                     size = train_num, replace = False))
    test_imgs = [i for i in images if i not in train_imgs]
    assert (len(images) == (len(train_imgs) + len(test_imgs))), 'Images must be in either training or testing set'
    for i in train_imgs:
        os.system(f'cp {datadir + row["category"] + "/" + i} {datadir + "train/" + row["category"] + "/" + i}')
    for i in test_imgs:
        os.system(f'cp {datadir + row["category"] + "/" + i} {datadir + "test/" + row["category"] + "/" + i}')
