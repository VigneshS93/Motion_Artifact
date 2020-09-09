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
from models import art_rem
from torch.utils.data import DataLoader
from dataset import dataset_loader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scd 
# from dataloader import load_data as data_loader


#Pass the arguments
parser = argparse.ArgumentParser(description="art_rem")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--num_epochs", type=int, default=200, help="Number of training epochs")
parser.add_argument("--decay_step", type=int, default=25, help="The step at which the learning rate should drop")
parser.add_argument("--lr_decay", type=float, default=0.7, help='Rate at which the learning rate should drop')
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--data_dir", type=str, default=" ", help='path of data')
parser.add_argument("--log_dir", type=str, default=" ", help='path of log files')
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--write_freq", type=int, default=10, help="Step for saving Checkpoint")

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# load the training data set
input_set, groundTruth_set = dataset_loader(opt.data_dir)
train_set=[]
for i in range(len(input_set)):
  train_set.append([input_set[i], groundTruth_set[i]])
trainLoader = DataLoader(dataset=train_set, num_workers=0, batch_size=opt.batchSize, shuffle=True, pin_memory=True)
#Convert the panda dataframe to Torch tensor

# Define the network 
net = network()
# Define the loss function
def my_loss(original, predicted):
  h, w, _ = np.shape(original)
  loss = (1/(h*w))*(np.sqrt(np.sum(np.square(abs(np.subtract(original,predicted))))))
  return loss
iters = -1
# Define the optimizer
lr_clip = 1e-5
optimizer = optim.Adam(model.parameters(), lr=opt.lr)
lr_lbmd = lambda it: max(opt.lr_decay ** (int(it * opt.batch_size / opt.decay_step)),lr_clip / opt.lr,)
lr_scheduler = lr_scd.LambdaLR(optimizer,lr_lambda=lr_lmbd, last_epoch=iters)

#Define the log directory for checkpoints
if os.path.exists(opt.log_dir) is not True:
  os.makedirs(opt.log_dir)

checkpoints_dir = os.path.join(opt.log_dir, "checkpoints")

if os.path.exists(checkpoints_dir) is not True:
  os.mkdir(checkpoints_dir)

#Load status from checkpoint 
log_open_mode = 'w'
start_epoch = 0
if opt.checkpoint is not None:
    fname = os.path.join(checkpoints_dir, opt.checkpoint)
    start_epoch, iters = checkpoint_util.load_checkpoint(model_3d=model, optimizer=optimizer, filename=fname)
    start_epoch += 1
    log_open_mode = 'a'

log = LogUtils(os.path.join(opt.log_dir, 'logfile'), log_open_mode)
log.write('Supervised learning for motion artifact reduction\n')
log.write_args(opt)
iters = max(iters,0)
# Train the network using the training dataset
for epoch_num in range(start_epoch, opt.num_epochs):
  for data in iter(trainLoader):
    if lr_scheduler is not None:
      lr_scheduler.step(iters)
    optimizer.zero_grad()
    for param_group in optimizer.param_groups:
          param_group["lr"] = current_lr
      print('learning rate %f' % current_lr)
    inp_PM, gt = next(trainLoader)
    # forward + backward + optimize
    outputs = net(img)
    loss = my_loss(outputs, img_label)
    loss.backward()
    optimizer.step()
    iters += 1
  if opt.write_freq != -1 and (epoch_num + 1) % opt.write_freq is 0:
    fname = os.path.join(checkpoints_dir, 'checkpoint_{}'.format(epoch_i))
    checkpoint_util.save_checkpoint(filename=fname, model_3d=model, optimizer=optimizer, iters=iters, epoch=epoch_num)

print('Finished Training')

# save the trained model
PATH = './motion_artifact.pth'
torch.save(net.state_dict(), PATH)
