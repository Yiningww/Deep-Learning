import torchvision
from torch import nn
import torch
train_data = torchvision.datasets.CIFAR10("dataset", train = False,
                                           download = True, transform = torchvision.transforms.ToTensor())

vgg16_false = torchvision.models.vgg16(pretrained = False) # Linear(in_features=4096, out_features=1000, bias=True)
vgg16_true = torchvision.models.vgg16(pretrained = True)

vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
vgg16_false.classifier[6] = nn.Linear(4096, 10)
vgg16 = torchvision.models.vgg16(pretrained = False)
# Preserve Method 1: Model architecture + Model Parameter
torch.save(vgg16, "vgg16_method1.pth")


# Preserve Method 2: Model Parameter (recommended)
torch.save(vgg16.state_dict(), "vgg16_method2.pth")


# Trap 1
class YN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3)

    def forward(self, input):
        x = self.conv1(x)
        return x

yn = YN()
torch.save(yn, "yn_method1.pth")