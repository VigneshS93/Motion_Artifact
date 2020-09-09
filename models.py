#model
import torch
import torch.nn as nn

class art_rem(torch.nn.Module):
    def __init__(self, inp_ch):
        super(network, self).__init__()
        ks = 3
        pad = 1
        out_ch_1 = 10
        out_ch_2 = 20
        out_ch_3 = 30
        self.conv1 = nn.Conv2d(inp_ch, out_ch_1, kernel_size=ks, stride=1, padding=pad)
        self.conv2 = nn.Conv2d(out_ch_1, out_ch_2, kernel_size=ks, stride=1, padding=pad)
        self.conv3 = nn.Conv2d(out_ch_2, out_ch_3, kernel_size=ks, stride=1, padding=pad)
        self.mpool1 = nn.MaxPool2d(2, stride=2,return_indices=True)
        self.conv4 = nn.Conv2d(out_ch_3, out_ch_3, kernel_size=ks, stride=1, padding=pad)
        self.conv5 = nn.Conv2d(out_ch_3, out_ch_3, kernel_size=ks, stride=1, padding=pad)
        self.mpool2 = nn.MaxUnPool2d(2, stride=2)
        self.conv6 = nn.Conv2d(out_ch_3, out_ch_2, kernel_size=ks, stride=1, padding=pad)
        self.conv7 = nn.Conv2d(out_ch_2, out_ch_1, kernel_size=ks, stride=1, padding=pad)
        self.conv8 = nn.Conv2d(out_ch_1, inp_ch, kernel_size=ks, stride=1, padding=pad)
     
        
       
    def forward(self, X):
        h = F.relu(self.conv1(X))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h, indices = F.relu(self.mpool1(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.relu(self.mpool2(h, indices))
        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        h = F.relu(self.conv8(h))
        
        return h