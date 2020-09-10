import torch
from torchvision import transforms
import pandas as pd
import os
import glob
import os.path
import numpy as np
import random
import h5py
import cv2
import glob
import torch.utils.data as udata

def dataset_loader(data_dir):
      extension = 'csv'
      # Read the input 
      directory=data_dir+str("/input")
      os.chdir(directory)
      files = [i for i in glob.glob('*.{}'.format(extension))]
      train_data = []
      for f in files:
            inp = pd.read_csv(f,header=None)
            train_data.append(inp)
      #Read the groundtruth data
      directory=data_dir+str("/groundTruth")
      os.chdir(directory)
      files = [i for i in glob.glob('*.{}'.format(extension))]
      gt_data = []
      for f in files:
            inp = pd.read_csv(f,header=None)
            gt_data.append(inp)
      return train_data, gt_data

def normalize(data):
    return data/255.
