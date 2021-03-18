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
from PIL import Image

def dataset_loader(data_dir):
      extension = 'csv'
      # Read the input 
      directory=data_dir+str("/input")
      os.chdir(directory)
      files = [i for i in glob.glob('*.{}'.format(extension))]
      files = sorted(files)
      filename = []
      train_data = []
      for f in files:
            inp = pd.read_csv(f,header=None)
            filename_temp = str(f)
            filename.append(filename_temp)
            train_data.append(inp)
      #Read the groundtruth data
      directory=data_dir+str("/groundTruth")
      os.chdir(directory)
      files = [i for i in glob.glob('*.{}'.format(extension))]
      files = sorted(files)
      gt_data = []
      for f in files:
            inp = pd.read_csv(f,header=None)
            gt_data.append(inp)
      #Read the mask
      directory=data_dir+str("/mask")
      os.chdir(directory)
      files = [i for i in glob.glob('*.{}'.format(extension))]
      files = sorted(files)
      mask_data = []
      for f in files:
            inp = pd.read_csv(f,header=None)
            mask_data.append(inp)
      return train_data, gt_data, mask_data, filename

def normalizeData(data):
      norm = torch.zeros(np.shape(data))
      for bs in range(len(data)):
            temp = data[bs]           
            norm_temp = temp - torch.min(temp)
            norm_temp = norm_temp/torch.max(norm_temp)
            norm[bs] = norm_temp
            # temp = data[bs]
            # data_range = torch.max(temp) - torch.min(temp)
            # norm_temp = (temp - torch.min(temp))/data_range
            # norm_temp = 2 * norm_temp - 1
      # norm = torch.FloatTensor(np.array(norm))
      return norm

# data loader for the fringe to fringe network
def dataLoader_f2f(data_dir, h_or_l, start, end):
      # h_or_l : should either be 'h' or 'l'
      # Input data folder
      # moved fringe folder
      directory_m = data_dir + str("/input_") + h_or_l + "fringe" 
      # static fringe folder
      directory_s = data_dir + str("/input_static")
      
      # Label data folder
      directory_label = data_dir + str("/label_static_") + h_or_l 

      # Mask data folder
      directory_mask = data_dir + str("/mask")

      train_data = []
      gt_data = []
      mask_data = []
      filename = []

      for i in range(start, end + 1):
            # Read input: moved + static
            fringe_moved = np.asarray(Image.open(directory_m + "/input_" + h_or_l + "fringe_{:04d}.png".format(i)), dtype=np.uint8)
            fringe_static = np.asarray(Image.open(directory_s + "/input_static_{:04d}.png".format(i)), dtype=np.uint8)
            temp = np.zeros((2, fringe_moved.shape[0], fringe_moved.shape[1]), 'uint8')
            temp[1, ...] = fringe_moved
            temp[0, ...] = fringe_static
            train_data.append(temp) 

            # Read ground truth: static
            temp = np.asarray(Image.open(directory_label + "/label_static_" + h_or_l + "_{:04d}.png".format(i)), dtype=np.uint8)
            gt_data.append(temp)

            # Read mask
            temp = pd.read_csv(directory_mask + "/mask_{:04d}.csv".format(i), header=None)
            mask_data.append(temp)

      return train_data, gt_data, mask_data

# data loader for the wrapped phase to absolute phase Unet
def dataLoader_wp2p_unet(data_dir, start, end):
      # h_or_l : should either be 'h' or 'l'
      # Input data folder
      # wrapped high-frequency phase map
      directory_wph = data_dir + str("/input_wph")
      # wrapped low-frequency phase map
      directory_wpl = data_dir + str("/input_wpl")
      
      # Label data folder
      directory_label = data_dir + str("/label_uph")

      # Mask data folder
      directory_mask = data_dir + str("/mask")

      train_data = []
      gt_data = []
      mask_data = []
      filename = []

      for i in range(start, end + 1):
            #if 176 <= i <= 200 or 226 <= i <= 250 or 301 <= i <= 325:
              #continue
            
            #if 36 <= i <= 40:
              #continue  
            
            # Read input: wph + wpl
            wph = np.genfromtxt(directory_wph + "/input_wph_{:04d}.csv".format(i), delimiter=',')
            wpl = np.genfromtxt(directory_wpl + "/input_wpl_{:04d}.csv".format(i), delimiter=',')
            temp = np.zeros((2, wph.shape[0], wpl.shape[1]), 'float')
            temp[1, ...] = wpl
            temp[0, ...] = wph
            # temp = temp[:, 1:512, 16:527]
            train_data.append(temp) 

            # Read ground truth: absolute phase
            temp = pd.read_csv(directory_label + "/label_uph_{:04d}.csv".format(i), header=None)
            # temp = temp[1:512, 16:527]
            gt_data.append(temp)

            # Read mask
            temp = pd.read_csv(directory_mask + "/mask_{:04d}.csv".format(i), header=None)
            # temp = temp[1:512, 16:527]
            mask_data.append(temp)

      return train_data, gt_data, mask_data
      
# data loader for the wrapped phase to absolute phase Unet, uses the index of object, angle, and motion (oam)
def dataLoader_wp2p_unet_oam(data_dir, start, end, num_of_ang, num_of_motion):
      # h_or_l : should either be 'h' or 'l'
      # Input data folder
      # wrapped high-frequency phase map
      directory_wph = data_dir + str("/input_wph")
      # wrapped low-frequency phase map
      directory_wpl = data_dir + str("/input_wpl")
      
      # Label data folder
      directory_label = data_dir + str("/label_uph")

      # Mask data folder
      directory_mask = data_dir + str("/mask")

      train_data = []
      gt_data = []
      mask_data = []
      filename = []

      for i in range(start, end + 1):
        if i in [8,  10,  13, 44]: # bird, donkey, simpson, penguin having unwrapping error
            continue
        for j in range(1, num_of_ang + 1):
            if i == 1 and j in [2, 4]:
                continue
            if i == 2 and j in [2, 3, 5]:
                continue
            if i == 3 and j in [4]:
                continue
            if i == 7 and j in [5]:
                continue
            if i == 9 and j in [5]:
                continue
            if i == 11 and j in [2, 5]:
                continue
            if i == 12 and j in [2,3,4,5]:
                continue
            if i == 14 and j in [2, 3, 5]:
                continue
            if i == 15 and j in [4, 5]:
                continue
            if i == 16 and j in [5]:
                continue
            if i == 20 and j in [2]:
                continue
            if i == 9 and j in [5]:
                continue
            if i == 21 and j in [5]:
                continue
            if i == 22 and j in [5]:
                continue
            if i == 23 and j in [4, 5]:
                continue
            if i == 24 and j in [2,4,5]:
                continue
            if i == 29 and j in [4, 5]:
                continue
            if i == 31 and j in [4, 5]:
                continue
            if i == 36 and j in [2, 3]:
                continue
            if i == 37 and j in [4, 5]:
                continue
            if i == 38 and j in [1,4]:
                continue
            if i == 40 and j in [4, 5]:
                continue
            if i == 42 and j in [4, 5]:
                continue
            if i == 43 and j in [5]:
                continue
             
            for k in range(1, num_of_motion + 1):
              # Read input: wph + wpl
              wph = np.genfromtxt(directory_wph + "/input_wph_{}_{}_{}.csv".format(i, j, k), delimiter=',')
              wpl = np.genfromtxt(directory_wpl + "/input_wpl_{}_{}_{}.csv".format(i, j, k), delimiter=',')
              temp = np.zeros((2, wph.shape[0], wpl.shape[1]), 'float')
              temp[1, ...] = wpl
              temp[0, ...] = wph
              # temp = temp[:, 1:512, 16:527]
              train_data.append(temp) 

              # Read ground truth: absolute phase
              temp = pd.read_csv(directory_label + "/label_uph_{}_{}_{}.csv".format(i, j, k), header=None)
              # temp = temp[1:512, 16:527]
              gt_data.append(temp)

              # Read mask
              temp = pd.read_csv(directory_mask + "/mask_{}_{}_{}.csv".format(i, j, k), header=None)
              # temp = temp[1:512, 16:527]
              mask_data.append(temp)
              
              # Save filenames
              filename.append("{}_{}_{}".format(i, j, k))

      return train_data, gt_data, mask_data, filename

# data loader for the wrapped high phase and unwrapped low phase to absolute phase Unet, uses the index of object, angle, and motion (oam)
def dataLoader_whul2p_unet_oam(data_dir, start, end, num_of_ang, num_of_motion):
      # h_or_l : should either be 'h' or 'l'
      # Input data folder
      # wrapped high-frequency phase map
      directory_wph = data_dir + str("/input_wph")
      # wrapped low-frequency phase map
      directory_wpl = data_dir + str("/input_upl")
      
      # Label data folder
      directory_label = data_dir + str("/label_uph")

      # Mask data folder
      directory_mask = data_dir + str("/mask")

      train_data = []
      gt_data = []
      mask_data = []
      filename = []

      for i in range(start, end + 1):
        if i in [8,  10,  13, 44]: # bird, donkey, simpson, penguin having unwrapping error
            continue
        for j in range(1, num_of_ang + 1):
            if i == 1 and j in [2, 4]:
                continue
            if i == 2 and j in [2, 3, 5]:
                continue
            if i == 3 and j in [4]:
                continue
            if i == 7 and j in [5]:
                continue
            if i == 9 and j in [5]:
                continue
            if i == 11 and j in [2, 5]:
                continue
            if i == 12 and j in [2,3,4,5]:
                continue
            if i == 14 and j in [2, 3, 5]:
                continue
            if i == 15 and j in [4, 5]:
                continue
            if i == 16 and j in [5]:
                continue
            if i == 20 and j in [2]:
                continue
            if i == 9 and j in [5]:
                continue
            if i == 21 and j in [5]:
                continue
            if i == 22 and j in [5]:
                continue
            if i == 23 and j in [4, 5]:
                continue
            if i == 24 and j in [2,4,5]:
                continue
            if i == 29 and j in [4, 5]:
                continue
            if i == 31 and j in [4, 5]:
                continue
            if i == 36 and j in [2, 3]:
                continue
            if i == 37 and j in [4, 5]:
                continue
            if i == 38 and j in [1,4]:
                continue
            if i == 40 and j in [4, 5]:
                continue
            if i == 42 and j in [4, 5]:
                continue
            if i == 43 and j in [5]:
                continue
             
            for k in range(1, num_of_motion + 1):
              # Read input: wph + wpl
              wph = np.genfromtxt(directory_wph + "/input_wph_{}_{}_{}.csv".format(i, j, k), delimiter=',')
              wpl = np.genfromtxt(directory_wpl + "/input_upl_{}_{}_{}.csv".format(i, j, k), delimiter=',') # did not name the variable properly
              temp = np.zeros((2, wph.shape[0], wpl.shape[1]), 'float')
              temp[1, ...] = wpl
              temp[0, ...] = wph
              # temp = temp[:, 1:512, 16:527]
              train_data.append(temp) 

              # Read ground truth: absolute phase
              temp = pd.read_csv(directory_label + "/label_uph_{}_{}_{}.csv".format(i, j, k), header=None)
              # temp = temp[1:512, 16:527]
              gt_data.append(temp)

              # Read mask
              temp = pd.read_csv(directory_mask + "/mask_{}_{}_{}.csv".format(i, j, k), header=None)
              # temp = temp[1:512, 16:527]
              mask_data.append(temp)
              
              # Save filenames
              filename.append("{}_{}_{}".format(i, j, k))

      return train_data, gt_data, mask_data, filename

# data loader for the unwrapped high phase and unwrapped low phase to absolute phase, uses the index of object, angle, and motion (oam)
def dataLoader_uhul2p_unet_oam(data_dir, start, end, num_of_ang, num_of_motion):
      # h_or_l : should either be 'h' or 'l'
      # Input data folder
      # wrapped high-frequency phase map
      directory_wph = data_dir + str("/input_uph")
      # wrapped low-frequency phase map
      directory_wpl = data_dir + str("/input_upl")
      
      # Label data folder
      directory_label = data_dir + str("/label_uph")

      # Mask data folder
      directory_mask = data_dir + str("/mask")

      train_data = []
      gt_data = []
      mask_data = []
      filename = []

      for i in range(start, end + 1):
        if i in [8,  10,  13, 44]: # bird, donkey, simpson, penguin having unwrapping error
            continue
        for j in range(1, num_of_ang + 1):       
            for k in range(1, num_of_motion + 1):
              # Read input: uph + upl
              uph = np.genfromtxt(directory_wph + "/input_uph_{}_{}_{}.csv".format(i, j, k), delimiter=',')
              upl = np.genfromtxt(directory_wpl + "/input_upl_{}_{}_{}.csv".format(i, j, k), delimiter=',') # did not name the variable properly
              temp = np.zeros((2, uph.shape[0], upl.shape[1]), 'float')
              temp[1, ...] = upl
              temp[0, ...] = uph
              # temp = temp[:, 1:512, 16:527]
              train_data.append(temp) 

              # Read ground truth: absolute phase
              temp = pd.read_csv(directory_label + "/label_uph_{}_{}_{}.csv".format(i, j, k), header=None)
              # temp = temp[1:512, 16:527]
              gt_data.append(temp)

              # Read mask
              temp = pd.read_csv(directory_mask + "/mask_{}_{}_{}.csv".format(i, j, k), header=None)
              # temp = temp[1:512, 16:527]
              mask_data.append(temp)
              
              # Save filenames
              filename.append("{}_{}_{}".format(i, j, k))

      return train_data, gt_data, mask_data, filename
# data loader for the unwrapped high phase and wrapped high phase to absolute phase, uses the index of object, angle, and motion (oam)
def dataLoader_uhwh2p_unet_oam(data_dir, start, end, num_of_ang, num_of_motion):
      # h_or_l : should either be 'h' or 'l'
      # Input data folder
      # wrapped high-frequency phase map
      directory_uph = data_dir + str("/input_uph")
      # wrapped low-frequency phase map
      directory_wph = data_dir + str("/input_wph")
      
      # Label data folder
      directory_label = data_dir + str("/label_uph")

      # Mask data folder
      directory_mask = data_dir + str("/mask")

      train_data = []
      gt_data = []
      mask_data = []
      filename = []

      for i in range(start, end + 1):
        if i in [8,  10,  13, 44]: # bird, donkey, simpson, penguin having unwrapping error
            continue
        for j in range(1, num_of_ang + 1):       
            for k in range(1, num_of_motion + 1):
              # Read input: uph + upl
              uph = np.genfromtxt(directory_uph + "/input_uph_{}_{}_{}.csv".format(i, j, k), delimiter=',')
              wph = np.genfromtxt(directory_wph + "/input_wph_{}_{}_{}.csv".format(i, j, k), delimiter=',') 
              temp = np.zeros((2, uph.shape[0], wph.shape[1]), 'float')
              temp[1, ...] = wph
              temp[0, ...] = uph
              # temp = temp[:, 1:512, 16:527]
              train_data.append(temp) 

              # Read ground truth: absolute phase
              temp = pd.read_csv(directory_label + "/label_uph_{}_{}_{}.csv".format(i, j, k), header=None)
              # temp = temp[1:512, 16:527]
              gt_data.append(temp)

              # Read mask
              temp = pd.read_csv(directory_mask + "/mask_{}_{}_{}.csv".format(i, j, k), header=None)
              # temp = temp[1:512, 16:527]
              mask_data.append(temp)
              
              # Save filenames
              filename.append("{}_{}_{}".format(i, j, k))

      return train_data, gt_data, mask_data, filename
def dataLoader_whk2p_unet_oam(data_dir, start, end, num_of_ang, num_of_motion):
      # h_or_l : should either be 'h' or 'l'
      # Input data folder
      # wrapped high-frequency phase map
      directory_wph = data_dir + str("/input_wph")
      # wrapped low-frequency phase map
      directory_k = data_dir + str("/input_k")
      
      # Label data folder
      directory_label = data_dir + str("/label_uph")

      # Mask data folder
      directory_mask = data_dir + str("/mask")

      train_data = []
      gt_data = []
      mask_data = []
      filename = []

      for i in range(start, end + 1):
        if i in [8,  10,  13, 44]: # bird, donkey, simpson, penguin having unwrapping error
            continue
        for j in range(1, num_of_ang + 1):       
            for k in range(1, num_of_motion + 1):
              # Read input: uph + upl
              inp_k = np.genfromtxt(directory_k + "/input_k_{}_{}_{}.csv".format(i, j, k), delimiter=',')
              wph = np.genfromtxt(directory_wph + "/input_wph_{}_{}_{}.csv".format(i, j, k), delimiter=',') 
              temp = np.zeros((2, wph.shape[0], wph.shape[1]), 'float')
              temp[1, ...] = inp_k
              temp[0, ...] = wph
              # temp = temp[:, 1:512, 16:527]
              train_data.append(temp) 

              # Read ground truth: absolute phase
              temp = pd.read_csv(directory_label + "/label_uph_{}_{}_{}.csv".format(i, j, k), header=None)
              # temp = temp[1:512, 16:527]
              gt_data.append(temp)

              # Read mask
              temp = pd.read_csv(directory_mask + "/mask_{}_{}_{}.csv".format(i, j, k), header=None)
              # temp = temp[1:512, 16:527]
              mask_data.append(temp)
              
              # Save filenames
              filename.append("{}_{}_{}".format(i, j, k))

      return train_data, gt_data, mask_data, filename
def dataLoader_whuhul2p_unet_oam(data_dir, start, end, num_of_ang, num_of_motion):
      # h_or_l : should either be 'h' or 'l'
      # Input data folder
      # wrapped high-frequency phase map
      directory_wph = data_dir + str("/input_wph")
      # wrapped high-frequency phase map
      directory_uph = data_dir + str("/input_uph")
      # unwrapped low-frequency phase map
      directory_wpl = data_dir + str("/input_upl")
      
      # Label data folder
      directory_label = data_dir + str("/label_uph")

      # Mask data folder
      directory_mask = data_dir + str("/mask")

      train_data = []
      gt_data = []
      mask_data = []
      filename = []

      for i in range(start, end + 1):
        if i in [8,  10,  13, 44]: # bird, donkey, simpson, penguin having unwrapping error
            continue
        for j in range(1, num_of_ang + 1):
            for k in range(1, num_of_motion + 1):
              # Read input: wph + wpl
              wph = np.genfromtxt(directory_wph + "/input_wph_{}_{}_{}.csv".format(i, j, k), delimiter=',')
              uph = np.genfromtxt(directory_uph + "/input_uph_{}_{}_{}.csv".format(i, j, k), delimiter=',')
              wpl = np.genfromtxt(directory_wpl + "/input_upl_{}_{}_{}.csv".format(i, j, k), delimiter=',') # did not name the variable properly
              temp = np.zeros((3, wph.shape[0], wpl.shape[1]), 'float')
              temp[2, ...] = wpl
              temp[1, ...] = wph
              temp[0, ...] = uph
              # temp = temp[:, 1:512, 16:527]
              train_data.append(temp) 

              # Read ground truth: absolute phase
              temp = pd.read_csv(directory_label + "/label_uph_{}_{}_{}.csv".format(i, j, k), header=None)
              # temp = temp[1:512, 16:527]
              gt_data.append(temp)

              # Read mask
              temp = pd.read_csv(directory_mask + "/mask_{}_{}_{}.csv".format(i, j, k), header=None)
              # temp = temp[1:512, 16:527]
              mask_data.append(temp)
              
              # Save filenames
              filename.append("{}_{}_{}".format(i, j, k))

      return train_data, gt_data, mask_data, filename

def dataLoader_uhwhul2p_unet_oam_real(data_dir, start, end):
      # h_or_l : should either be 'h' or 'l'

      train_data = []
      mask_data = []
      filename = []
      
      for i in range(start, end + 1):
        # Read input: wph + wpl
        uph = np.genfromtxt(data_dir + "/input_uph_{}.csv".format(i), delimiter=',')
        upl = np.genfromtxt(data_dir + "/input_upl_{}.csv".format(i), delimiter=',')
        wph = np.genfromtxt(data_dir + "/input_wph_{}.csv".format(i), delimiter=',')
        temp = np.zeros((3, uph.shape[0], upl.shape[1]), 'float')
        temp[2, ...] = upl
        temp[1, ...] = wph
        temp[0, ...] = uph
        # temp = temp[:, 1:512, 16:527]
        train_data.append(temp) 

        # Read mask
        temp = pd.read_csv(data_dir + "/mask_{}.csv".format(i), header=None)
        # temp = temp[1:512, 16:527]
        mask_data.append(temp)
              
        # Save filenames
        filename.append("{}".format(i))

      return train_data, mask_data, filename

def dataLoader_uhwh2p_unet_oam_real(data_dir, start, end):
      # h_or_l : should either be 'h' or 'l'

      train_data = []
      mask_data = []
      filename = []
      
      for i in range(start, end + 1):
        # Read input: wph + wpl
        uph = np.genfromtxt(data_dir + "/input_uph_{}.csv".format(i), delimiter=',')
        # upl = np.genfromtxt(data_dir + "/input_upl_{}.csv".format(i), delimiter=',')
        wph = np.genfromtxt(data_dir + "/input_wph_{}.csv".format(i), delimiter=',')
        temp = np.zeros((2, uph.shape[0], wph.shape[1]), 'float')
        # temp[2, ...] = upl
        temp[1, ...] = wph
        temp[0, ...] = uph
        # temp = temp[:, 1:512, 16:527]
        train_data.append(temp) 

        # Read mask
        temp = pd.read_csv(data_dir + "/mask_{}.csv".format(i), header=None)
        # temp = temp[1:512, 16:527]
        mask_data.append(temp)
              
        # Save filenames
        filename.append("{}".format(i))

      return train_data, mask_data, filename


def dataLoader_wp2p_unet_oam_real(data_dir, start, end, num_of_ang, num_of_motion):
      # h_or_l : should either be 'h' or 'l'
      # Input data folder
      # wrapped high-frequency phase map
      directory_wph = data_dir + str("/input_wph")
      # wrapped low-frequency phase map
      directory_wpl = data_dir + str("/input_wpl")

      # Mask data folder
      directory_mask = data_dir + str("/mask")

      train_data = []
      mask_data = []
      filename = []
      
      for i in range(start, end + 1):
        for j in range(1, num_of_ang + 1):
            for k in range(1, num_of_motion + 1):
              # Read input: wph + wpl
              wph = np.genfromtxt(directory_wph + "/input_wph_{}_{}_{}.csv".format(i, j, k), delimiter=',')
              wpl = np.genfromtxt(directory_wpl + "/input_wpl_{}_{}_{}.csv".format(i, j, k), delimiter=',')
              temp = np.zeros((2, wph.shape[0], wpl.shape[1]), 'float')
              temp[1, ...] = wpl
              temp[0, ...] = wph
              # temp = temp[:, 1:512, 16:527]
              train_data.append(temp) 

              # Read mask
              temp = pd.read_csv(directory_mask + "/mask_{}_{}_{}.csv".format(i, j, k), header=None)
              # temp = temp[1:512, 16:527]
              mask_data.append(temp)
              
              # Save filenames
              filename.append("{}_{}_{}".format(i, j, k))

      return train_data, mask_data, filename



def dataLoader_p2p_oam(data_dir, start, end, num_of_ang, num_of_motion):
      # Input data folder
      # unwrapped phase map
      directory_uph = data_dir + str("/input_uph")
      
      # Label data folder
      directory_label = data_dir + str("/label_uph")

      # Mask data folder
      directory_mask = data_dir + str("/mask")

      train_data = []
      gt_data = []
      mask_data = []
      filenames = []

      for i in range(start, end + 1):
        if i in [8,  10,  13, 44]: # bird, donkey, simpson, penguin having unwrapping error
            continue
        for j in range(1, num_of_ang + 1):
            if i == 1 and j in [2, 4]:
                continue
            if i == 2 and j in [2, 3, 5]:
                continue
            if i == 3 and j in [4]:
                continue
            if i == 7 and j in [5]:
                continue
            if i == 9 and j in [5]:
                continue
            if i == 11 and j in [2, 5]:
                continue
            if i == 12 and j in [2,3,4,5]:
                continue
            if i == 14 and j in [2, 3, 5]:
                continue
            if i == 15 and j in [4, 5]:
                continue
            if i == 16 and j in [5]:
                continue
            if i == 20 and j in [2]:
                continue
            if i == 9 and j in [5]:
                continue
            if i == 21 and j in [5]:
                continue
            if i == 22 and j in [5]:
                continue
            if i == 23 and j in [4, 5]:
                continue
            if i == 24 and j in [2,4,5]:
                continue
            if i == 29 and j in [4, 5]:
                continue
            if i == 31 and j in [4, 5]:
                continue
            if i == 36 and j in [2, 3]:
                continue
            if i == 37 and j in [4, 5]:
                continue
            if i == 38 and j in [1,4]:
                continue
            if i == 40 and j in [4, 5]:
                continue
            if i == 42 and j in [4, 5]:
                continue
            if i == 43 and j in [5]:
                continue
             
            for k in range(1, num_of_motion + 1):
              # Read input: wph + wpl
              # Read input: uph
              temp = np.genfromtxt(directory_uph + "/input_uph_{}_{}_{}.csv".format(i, j, k), delimiter=',')
              #temp = temp[1:513, 16:528]
              train_data.append(temp) 

              # Read ground truth: absolute phase
              temp = pd.read_csv(directory_label + "/label_uph_{}_{}_{}.csv".format(i, j, k), header=None)
              # temp = temp[1:512, 16:527]
              gt_data.append(temp)

              # Read mask
              temp = pd.read_csv(directory_mask + "/mask_{}_{}_{}.csv".format(i, j, k), header=None)
              # temp = temp[1:512, 16:527]
              mask_data.append(temp)
              
              # Save filenames
              filenames.append("{}_{}_{}".format(i, j, k))

      return train_data, gt_data, mask_data, filenames

def dataLoader_f2wp_unet(data_dir, h_or_l, start, end):
      # Input data folder
      # unwrapped phase map
      directory_f = data_dir + str("/input_") + h_or_l + "_f123"
      
      # Label data folder
      directory_label = data_dir + str("/label_wp") + h_or_l

      # Mask data folder
      directory_mask = data_dir + str("/mask")

      train_data = []
      gt_data = []
      mask_data = []
      filename = []

      for i in range(start, end + 1):
            # Read input: wph + wpl
            temp = np.asarray(Image.open(directory_f + "/input_" + h_or_l + "_f123_{:04d}.png".format(i)), dtype=np.uint8)
            temp = np.reshape(temp, (3, 514, 544))
            temp = temp[:, 1:513, 16:528]
            train_data.append(temp) 

            # Read ground truth: wrapped phase
            temp = np.genfromtxt(directory_label + "/label_wp" + h_or_l + "_{:04d}.csv".format(i), delimiter=',')
            temp = temp[1:513, 16:528]
            gt_data.append(temp)

            # Read mask
            temp = pd.np.genfromtxt(directory_mask + "/mask_{:04d}.csv".format(i), delimiter=',')
            temp = temp[1:513, 16:528]
            mask_data.append(temp)

      return train_data, gt_data, mask_data
