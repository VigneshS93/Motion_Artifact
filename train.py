import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# load the training data set
#dataiter = iter(trainData)
#images, labels = dataiter.next()

# Define the network
class network(torch.nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(3, 12, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(24, 36, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(36, 48, kernel_size=3, stride=1, padding=1)
        
        self.conv6 = nn.Conv2d(48, 36, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(36, 24, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(24, 12, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(12, 3, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1)
        
        
# average the pixels to get the required dimension
       
    def forward(self, X):
        h = F.relu(self.conv1(X))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        
        h = F.relu(self.conv6(X))
        h = F.relu(self.conv7(h))
        h = F.relu(self.conv8(h))
        h = F.relu(self.conv9(h))
        h = F.relu(self.conv10(h))
        
        return h
  
net = network()
# Define the loss function
def my_loss(original, predicted):
  h, w, _ = np.shape(original)
  loss = (1/(h*w))*(np.sqrt(np.sum(np.square(abs(np.subtract(original,predicted))))))
  return loss

# Compute loss using the loss function
#criterion = my_loss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


train_groundtruth_path = 'sample_data/train/groundtruth'
train_noisy_path = 'sample_data/train/noisy'

test_groundtruth_path = 'sample_data/test/groundtruth'
test_noisy_path = 'sample_data/test/noisy'

train_label = os.listdir(train_groundtruth_path)
train_input = os.listdir(train_noisy_path)

test_label = os.listdir(test_groundtruth_path)
test_input = os.listdir(test_noisy_path)


# Train the network using the training dataset
for epoch in range(20):  
    
    for im in train_label:

      image_path = os.path.join(train_noisy_path,im)
      img = torch.tensor(cv2.imread(image_path))

      label_path = os.path.join(train_groundtruth_path,im)
      img_label = torch.tensor(cv2.imread(label_path))
     
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = net(img)
      loss = my_loss(outputs, img_label)
      loss.backward()
      optimizer.step()

print('Finished Training')

# save the trained model
PATH = './motion_artifact.pth'
torch.save(net.state_dict(), PATH)

# Load test data

# Test the network for the entire test data set
with torch.no_grad():

  for im in test_label:

      image_path = os.path.join(test_noisy_path,im)
      img = torch.tensor(cv2.imread(image_path))

      label_path = os.path.join(test_groundtruth_path,im)
      img_label = cv2.imread(label_path)
     
      # forward + backward + optimize
      outputs = net(img)

       
# Visualize the output images
_, predicted = (outputs,1)
