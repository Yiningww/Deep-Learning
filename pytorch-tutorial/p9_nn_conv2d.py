import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("dataset", train = False, 
                                       transform = torchvision.transforms.ToTensor(),
                                       download = True)
# each data: (img, target) -> img: torch.Size([3, 32, 32])
dataloader = DataLoader(dataset, batch_size = 64)
img_0, target_0 = dataset[0]

class YN(nn.Module):
    def __init__(self):
        super(YN, self).__init__()
        self.conv1 = Conv2d(in_channels = 3, out_channels = 6, kernel_size = 3, stride = 1, padding = 0)
    
    def forward(self, x):
        x = self.conv1(x)
        return x
    
yn = YN()
print(yn)

writer = SummaryWriter("p9")
# for epoch in range(2):
step = 0
for data in dataloader:
    imgs, targets = data # imgs.shape: torch.Size([64, 3, 32, 32])
    output = yn(imgs) # output.shape: torch.Size([64, 6, 30, 30]) -> [xxx, 3, 30, 30]
    writer.add_images("input", imgs, step)
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)
    step = step + 1

writer.close()



