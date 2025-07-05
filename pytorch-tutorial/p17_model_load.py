import torch
import torchvision
import torch.nn as nn

vgg16 = torchvision.models.vgg16(pretrained = False)

# Preserve Method 1: Model architecture + Model Parameter
model1 = torch.load("vgg16_method1.pth")
# print(model1)

# Preserve Method 2: Model Parameter (recommended)
model2 = torch.load("vgg16_method2.pth")
vgg16.load_state_dict(model2)

print(vgg16)
p
# ls -all
# -rw-r--r--@  1 wangyining  staff  537227470 Mar 14 00:51 vgg16_method1.pth
# -rw-r--r--@  1 wangyining  staff  553441550 Mar 14 00:54 vgg16_method2.pth

# Trap 1
class YN(nn.Module): # Can't get attribute 'YN' on <module '__main__' from 'p17_model_load.py'>
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3)

    def forward(self, input):
        x = self.conv1(x)
        return x
    

model = torch.load("yn_method1.pth") # you have to include the model architecture here
print(model)