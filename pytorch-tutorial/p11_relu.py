import torch 
from torch import nn
from torch.nn import ReLU, Sigmoid
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("dataset", train = False, download = True,
                                       transform = torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size = 64)


input = torch.tensor([[1, -0.5],
                      [-1, 3]])

input2 = torch.reshape(input, (-1, 1, 2, 2))
# print(input.shape)

class YN(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()
    
    def forward(self, x):
        output = self.sigmoid1(x)
        return output

yn = YN()
print(yn(input2)) #input also ok

writer = SummaryWriter("relu")
step = 0
for data in dataloader:
    imgs, targets = data # imgs.shape: torch.Size([64, 3, 32, 32]) = output.shape
    output = yn(imgs) # Equal to output = nn.Sigmoid()(imgs)
    writer.add_images("input", imgs, step)
    writer.add_images("output", output, step)
    step += 1

writer.close()