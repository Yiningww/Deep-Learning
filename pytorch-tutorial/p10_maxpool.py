import torch 
from torch import nn
from torch.nn import MaxPool2d
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("dataset", train = False, download = True,
                                       transform = torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size = 64)

input = torch.tensor([[1, 2, 0, 3, 1], # torch.Size([5, 5])
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype = torch.float32)


input = torch.reshape(input, (-1, 1, 5, 5)) # input.shape: torch.Size([1, 1, 5, 5])

class YN(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = MaxPool2d(kernel_size = 3, ceil_mode = True)
    
    def forward(self, input):
        output = self.maxpool(input)
        # output = MaxPool2d(kernel_size = 3, ceil_mode = True)(input)
        return output
    
yn = YN()
# print(yn(input))

writer = SummaryWriter("p10")
step = 0
for data in dataloader:
    imgs, targets = data # imgs.shape: torch.Size([64, 3, 32, 32])
    output = yn(imgs) # output.shape: torch.Size([64, 3, 11, 11])
    writer.add_images("imgs", imgs, step)
    writer.add_images("outputs", output, step)
    step += 1

writer.close()