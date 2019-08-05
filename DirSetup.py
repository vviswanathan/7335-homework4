# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 09:02:31 2019

@author: vivek
"""

import os, os.path
import shutil
import math
import time

print(time.strftime("%Y-%m-%d %H:%M:%S"))

basedir = 'Data/'
datadir = basedir + '101_ObjectCategories/'
traindir = basedir + 'train/'
validdir = basedir + 'valid/'
testdir = basedir + 'test/'

shutil.rmtree(traindir)
shutil.rmtree(validdir)
shutil.rmtree(testdir)

shutil.copytree(datadir, traindir)
shutil.copytree(datadir, validdir)
shutil.copytree(datadir, testdir)

for d in os.listdir(datadir):
    
    d=d+'/'
    file_cnt = len([name for name in os.listdir(datadir+d) if os.path.isfile(os.path.join(datadir+d,name))])
    train_cnt = math.floor(file_cnt*0.5)
    test_cnt = math.ceil(file_cnt*0.25)
    valid_cnt = file_cnt-test_cnt-train_cnt
    
    print(d,file_cnt,train_cnt,test_cnt,valid_cnt)
    temp_num = 0
    for f in os.listdir(datadir+d):
        temp_num = temp_num + 1
        if temp_num <= train_cnt:
            os.unlink(validdir+d+f)
            os.unlink(testdir+d+f)
        elif temp_num <= test_cnt+train_cnt:
            os.unlink(validdir+d+f)
            os.unlink(traindir+d+f)
        else:
            os.unlink(traindir+d+f)
            os.unlink(testdir+d+f)
            
#    print(len([name for name in os.listdir(traindir+d) if os.path.isfile(os.path.join(traindir+d,name))]))
#    print(len([name for name in os.listdir(testdir+d) if os.path.isfile(os.path.join(testdir+d,name))]))
#    print(len([name for name in os.listdir(validdir+d) if os.path.isfile(os.path.join(validdir+d,name))]))

print(time.strftime("%Y-%m-%d %H:%M:%S"))