import torch
from torch import nn

class YN(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input): # input: tensor(1.) self:YN()
        output = input + 1 # output: tensor(2.)
        return output
    

yn = YN()
x = torch.tensor(1.0)
output = yn(x)
print(output)