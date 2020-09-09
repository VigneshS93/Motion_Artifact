import torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import os
import glob

def dataset_loader(data_dir):
      extension = 'csv'
      # Read the input data
      os.chdir("data_dir/input")
      files = [i for i in glob.glob('*.{}'.format(extension))]
      train_data = pd.concat([pd.read_csv(f) for f in files])
      #Read the groundtruth data
      os.chdir("data_dir/groundTruth")
      files = [i for i in glob.glob('*.{}'.format(extension))]
      gt_data = pd.concat([pd.read_csv(f) for f in files])
      return train_data, gt_data


# class PhaseMapDataset(Dataset):
#       def get_data(path):
#             inp = pd.read_csv(path)
#             return inp.as_matrix()
#   'Characterizes a dataset for PyTorch'
#   def __init__(self, data_dir, labels):
#         'Initialization'
#         self.labels = labels
#         self.list_IDs = list_IDs

#   def __len__(self):
#         'Denotes the total number of samples'
#         return len(self.list_IDs)

#   def __getitem__(self, index):
#         'Generates one sample of data'
#         # Select sample
#         ID = self.list_IDs[index]

#         # Load data and get label
#         inp_img = torch.load(data_dir + original + '.png')
#         label = torch.load(data_dir + groundTruth + '.png')

#         return inp_img, label