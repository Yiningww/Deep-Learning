from torch import nn
import torch

class YN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 5, stride = 1, padding = 2),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(), # 64*4*4
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )
    def forward(self, input):
        x = self.model1(input)
        return x
    

if __name__ == '__main__':
    yn = YN()
    input = torch.ones((64, 3, 32, 32))
    output = yn(input)
    print(output.shape)