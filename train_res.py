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
from datas import dataset_loader, dataLoader_whul2p_unet_oam, dataLoader_uhul2p_unet_oam, dataLoader_whuhul2p_unet_oam
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scd 
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
parser.add_argument("--lr_decay", type=float, default=0.5, help='Rate at which the learning rate should drop')
parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
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


# load the training data set
# input_set, groundTruth_set, mask = dataset_loader(opt.data_dir)
# input_set = torch.FloatTensor(np.array(input_set))
# groundTruth_set = torch.FloatTensor(np.array(groundTruth_set))
# mask = torch.FloatTensor(np.array(mask))
# mask = mask/255
# norm_input = normalizeData(input_set)
# norm_gt = normalizeData(groundTruth_set)
input_set, groundTruth_set, mask, filenames = dataLoader_uhul2p_unet_oam(opt.data_dir, opt.start_id, opt.end_id, opt.num_of_ang, opt.num_of_motion)
input_set = torch.FloatTensor(np.array(input_set))
groundTruth_set = torch.FloatTensor(np.array(groundTruth_set))
mask = torch.FloatTensor(np.array(mask))
train_set=[]
for i in range(len(input_set)):
  # train_set.append([input_set[i], norm_input[i], groundTruth_set[i], norm_gt[i], mask[i]])
  train_set.append([input_set[i], groundTruth_set[i], mask[i], filenames[i]])
num_workers = len(np.fromstring(opt.gpu_no, dtype=int, sep=','))
trainLoader = DataLoader(dataset=train_set, num_workers=num_workers, batch_size=opt.batchSize, shuffle=True, pin_memory=True)

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
model = nn.DataParallel(model) # For using multiple GPUs

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

#Load status from checkpoint 
log_open_mode = 'w'
start_epoch = 0
if opt.checkpoint is not None:
    fname = os.path.join(checkpoints_dir, opt.checkpoint)
    start_epoch, iters = checkpoint_util.load_checkpoint(model_3d=model, optimizer=optimizer, filename=fname)
    start_epoch += 1
    log_open_mode = 'a'

log = LogUtils(os.path.join(opt.log_dir, 'logfile'), log_open_mode)
log.write('Supervised learning for motion artifact reduction - Training\n')
log.write_args(opt)
lr_scheduler = lr_scd.StepLR(optimizer, step_size=opt.decay_step, gamma=opt.lr_decay)
iters = max(iters,0)
reg = 1e-7
# Train the network on the training dataset
for epoch_num in range(start_epoch, opt.num_epochs):
  trainData = iter(trainLoader)
  ave_loss = 0
  count = 0
  for data in iter(trainLoader):
    optimizer.zero_grad()
    if lr_scheduler is not None:
      lr_scheduler.step(iters)
    
    inp_PM, gt_PM, mask_PM, filename_PM = next(trainData)
    # inp_PM = torch.unsqueeze(inp_PM,1).cuda()
    inp_PM = inp_PM.cuda()
    gt_PM = torch.unsqueeze(gt_PM,1).cuda()
    mask_PM = torch.unsqueeze(mask_PM,1).cuda()
    output_PM = model(inp_PM)
    loss = ab_diff(mask_PM, output_PM, gt_PM)
    # loss = loss + rg_loss
    loss.backward()
    optimizer.step()
    iters += 1
    ave_loss += loss
    count += 1
    for ind in range(0,len(inp_PM)):
        out = output_PM[ind][0].detach().cpu().numpy()
        filename = opt.log_dir + str("/train_res/") + str(filename_PM[ind]) + str("_outputPM.csv")
        pd.DataFrame(out).to_csv(filename,header=False,index=False)
  lr_scheduler.get_last_lr()
  ave_loss /= count
  for param_group in optimizer.param_groups:
    print('\nTraining at Epoch %d with a learning rate of %f.' %(epoch_num, param_group["lr"]))
  if opt.write_freq != -1 and (epoch_num + 1) % opt.write_freq is 0:
    fname = os.path.join(checkpoints_dir, 'checkpoint_{}'.format(epoch_num))
    checkpoint_util.save_checkpoint(filename=fname, model_3d=model, optimizer=optimizer, iters=iters, epoch=epoch_num)

    # Write CSV files
    out = output_PM[0][0].detach().cpu().numpy()
    filename = opt.log_dir + str("/epoch_") + str(epoch_num) + str("_") + str(filename_PM[0]) + str("_outputPM.csv")
    pd.DataFrame(out).to_csv(filename,header=False,index=False)
  
  for ind in range(0,len(inp_PM)):
    out = output_PM[ind][0].detach().cpu().numpy()
    filename = opt.log_dir + str("/train_res/") + str(filename_PM[ind]) + str("_outputPM.csv")
    pd.DataFrame(out).to_csv(filename,header=False,index=False)
  # Log the results
  log.write('\nepoch no.: {0}, Average_train_loss:{1}'.format((epoch_num), ("%.8f" % ave_loss)))
  
print('Finished Training')


