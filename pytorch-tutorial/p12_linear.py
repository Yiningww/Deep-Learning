# VGG16 Model
import torch 
from torch import nn
from torch.nn import Linear, ReLU, Sigmoid
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


dataset = torchvision.datasets.CIFAR10("dataset", train = False, 
                                      transform = torchvision.transforms.ToTensor(),
                                      download = True)

dataloader = DataLoader(dataset, batch_size = 64)

class YN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(in_features = 196608, out_features = 10)
    
    def forward(self, input):
        output = self.linear1(input)
        return output

yn = YN()

for data in dataloader:
    imgs, targets = data # imgs.shape: torch.Size([64, 3, 32, 32])
    #input = torch.reshape(imgs, (1, 1, 1, -1)) # input.shape: torch.Size([1, 1, 1, 196608])
    input = torch.flatten(imgs) # input.shape: torch.Size([196608])
    output = yn(input) # output.shape: torch.Size([1, 1, 1, 10])/ torch.Size([10]) by flatten
