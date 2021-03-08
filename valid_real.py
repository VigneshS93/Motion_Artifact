import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd
from matplotlib.pyplot import imread
from models import art_rem1
from torch.utils.data import DataLoader
from datas import dataset_loader, dataLoader_whul2p_unet_oam, dataLoader_uhul2p_unet_oam, dataLoader_whuhul2p_unet_oam, dataLoader_uhul2p_unet_oam_real
import torch.optim as optim
# import torch.optim.lr_scheduler as lr_scd 
import sys
sys.path.append("..")
from utils.logutils import LogUtils
import utils.check_points_utils as checkpoint_util
from torch.autograd import Variable
from torchvision import transforms
from datas import normalizeData
# from dataloader import load_data as data_loader


#Pass the arguments
parser = argparse.ArgumentParser(description="art_rem1")
parser.add_argument("--batchSize", type=int, default=4, help="Training batch size")
parser.add_argument("--num_epochs", type=int, default=600, help="Number of training epochs")
parser.add_argument("--decay_step", type=int, default=100, help="The step at which the learning rate should drop")
# parser.add_argument("--lr_decay", type=float, default=0.5, help='Rate at which the learning rate should drop')
# parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
parser.add_argument("--data_dir", type=str, default=" ", help='path of data')
parser.add_argument("--log_dir", type=str, default=" ", help='path of log files')
parser.add_argument("--write_freq", type=int, default=50, help="Step for saving Checkpoint")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to start from")
parser.add_argument("--gpu_no", type=str, default="0", help="GPU number")
parser.add_argument("--input_channel", type=int, default=2, help="Input channels")
parser.add_argument("--start_id", type=int, default=1, help="Start data id")
parser.add_argument("--end_id", type=int, default=40, help="End data id")
parser.add_argument("--start_dev_id", type=int, default=0, help="Start data id for dev set")
parser.add_argument("--end_dev_id", type=int, default=0, help="End data id for dev set")
parser.add_argument("--num_of_ang", type=int, default=5, help="Number of angles that one object rotates for")
parser.add_argument("--num_of_motion", type=int, default=5, help="Number of motions that one object moves")

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_no


# load the validation data set

input_set, groundTruth_set, mask, filenames = dataLoader_uhul2p_unet_oam(opt.data_dir, opt.start_id, opt.end_id, opt.num_of_ang, opt.num_of_motion)
# input_set, mask, filenames = dataLoader_uhul2p_unet_oam_real(opt.data_dir, opt.start_id, opt.end_id)
input_set = torch.FloatTensor(np.array(input_set))
groundTruth_set = torch.FloatTensor(np.array(groundTruth_set))
mask = torch.FloatTensor(np.array(mask))
train_set=[]
for i in range(len(input_set)):
  # train_set.append([input_set[i], norm_input[i], groundTruth_set[i], norm_gt[i], mask[i]])
  train_set.append([input_set[i], groundTruth_set[i], mask[i], filenames[i]])
# num_workers = len(np.fromstring(opt.gpu_no, dtype=int, sep=','))
trainLoader = DataLoader(dataset=train_set, num_workers=0, batch_size=opt.batchSize, shuffle=True, pin_memory=True)

# Define the loss function
mse_loss = nn.MSELoss(reduction='mean')
def squared_diff(mask, output, groundTruth):
  sq_diff = torch.square(output - groundTruth)
  mask_sq_diff = torch.mul(mask,sq_diff)
  loss = torch.mean(mask_sq_diff)
  return loss
def ab_diff(mask, output, groundTruth):
  abs_diff = torch.abs(output - groundTruth)
  mask_abs_diff = torch.mul(mask,abs_diff)
  loss = torch.mean(mask_abs_diff)
  return loss
def percept_loss(output,regularization):
  reg_loss = regularization * (torch.sum(torch.abs(output[:, :, :, :-1] - output[:, :, :, 1:])) + torch.sum(torch.abs(output[:, :, :-1, :] - output[:, :, 1:, :])))
  return reg_loss
iters = -1

#Define the log directory for checkpoints
if os.path.exists(opt.log_dir) is not True:
  os.makedirs(opt.log_dir)

checkpoints_dir = os.path.join(opt.log_dir, "checkpoints")

if os.path.exists(checkpoints_dir) is not True:
  os.mkdir(checkpoints_dir)

# Load the model
model = art_rem1(opt.input_channel).cuda()
# model = nn.DataParallel(model) # For using multiple GPUs
# model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[rank],output_device=[rank])

#Load status from checkpoint 
log_open_mode = 'w'
checkpoint_util.load_checkpoint(model_3d=model, filename=opt.checkpoint)


log = LogUtils(os.path.join(opt.log_dir, 'logfile'), log_open_mode)
log.write('Supervised learning for motion artifact reduction - Validation\n')
log.write_args(opt)

if opt.checkpoint is None:
    print('Checkpoint is missing! Load the checkpoint to start the testing')
    sys.exit()

# Test the network on the trained model 
trainData = iter(trainLoader)
ave_loss = 0
count = 0
model.eval()
torch.no_grad()
for data in iter(trainLoader):
    inp_PM, gt_PM, mask_PM, filename_PM = next(trainData)
    # inp_PM = torch.unsqueeze(inp_PM,1).cuda()
    inp_PM = inp_PM.cuda()
    gt_PM = torch.unsqueeze(gt_PM,1).cuda()
    mask_PM = torch.unsqueeze(mask_PM,1).cuda()
    output_PM = model(inp_PM)
    loss = ab_diff(mask_PM, output_PM, gt_PM)
    print('\nThe loss is %f.' %(loss))
    ave_loss += loss
    count += 1
    # Write CSV files
    for ind in range(0,len(inp_PM)):
      out = output_PM[ind][0].detach().cpu().numpy()
      filename = opt.log_dir + str("/") + str(filename_PM[ind]) + str("_outputPM.csv")
      pd.DataFrame(out).to_csv(filename,header=False,index=False)
    ave_loss /= count

    

#   # Log the results
#   log.write('\nepoch no.: {0}, Average_train_loss:{1}'.format((epoch_num), ("%.8f" % ave_loss)))
  
print('Finished Testing')


