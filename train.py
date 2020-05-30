import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# define function to display the images
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
# load the training data set
dataiter = iter(trainData)
images, labels = dataiter.next()

# Define the network
class network(torch.nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(3, 12, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(24, 36, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(36, 48, kernel_size=3, stride=1, padding=1)
# average the pixels to get the required dimension
       
    def forward(self, X):
        h = F.relu(self.conv1(X))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        return h
  
net = network()
# Define the loss function
def my_loss(original, predicted):
  
  return loss

# Compute loss using the loss function
criterion = my_loss(y_original, y_pred)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train the network using the training dataset
for epoch in range(20):  
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# save the trained model
PATH = './motion_artifact.pth'
torch.save(net.state_dict(), PATH)

# Load test data
dataiter = iter(testloader)
images, labels = dataiter.next()

# Obtain the output for the test images
output = net(images);

# Visualize the output images
_, predicted = (outputs,1)
