import torch 
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, ReLU, Sigmoid
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class YN(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = Conv2d(3, 32, 5, padding = 2) # padding = 2
        # self.maxpool1 = MaxPool2d(2, stride = 2)
        # self.conv2 = Conv2d(32, 32, 5, padding = 2)
        # self.maxpool2 = MaxPool2d(2, stride = 2)
        # self.conv3 = Conv2d(32, 64, 5, padding = 2)
        # self.maxpool3 = MaxPool2d(2, stride = 2)
        # self.flatten = Flatten() # 64*4*4 = 1024
        # self.linear1 = Linear(1024, 64)
        # self.linear2 = Linear(64, 10)

        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding = 2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding = 2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding = 2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )
    
    def forward(self, x):
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x) # torch.Size([64, 1024])
        # x = self.linear1(x)
        # x = self.linear2(x)
        x = self.model1(x)
        return x
    
yn = YN()
print(yn)
input = torch.ones((64, 3, 32, 32))
output = yn(input) # torch.Size([64, 10]), 64 pictures
print(output.shape)

writer = SummaryWriter("p13")
writer.add_graph(yn, input)
writer.close()