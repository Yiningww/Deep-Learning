import torch 
from torch import nn
from torch.nn import L1Loss, Conv2d, MaxPool2d, Linear, Flatten
import torchvision
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("dataset", train = False, 
                                       transform = torchvision.transforms.ToTensor(),
                                       download = True)


dataloader = DataLoader(dataset, batch_size = 1)

inputs = torch.tensor([1, 2, 3], dtype = torch.float32)
targets = torch.tensor([1, 2, 5], dtype = torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3)) # tensor([[[[1.0, 2.0, 3.0]]]]), batch_size = 1, CHW = 1, 1, 3
targets = torch.reshape(targets, (1, 1, 1, 3)) # 1 batch size, 1 channel, 1 row, 3 column

loss = L1Loss(reduction = 'sum')
loss_mse = nn.MSELoss()
loss_cross_entropy = nn.CrossEntropyLoss()
result = loss(inputs, targets)
result_mes = loss_mse(inputs, targets)
result_ce = loss_cross_entropy(inputs, targets)


class YN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(3, 32, 5, padding = 2) # padding = 2
        self.maxpool1 = MaxPool2d(2, stride = 2)
        self.conv2 = Conv2d(32, 32, 5, padding = 2)
        self.maxpool2 = MaxPool2d(2, stride = 2)
        self.conv3 = Conv2d(32, 64, 5, padding = 2)
        self.maxpool3 = MaxPool2d(2, stride = 2)
        self.flatten = Flatten() # 64*4*4 = 1024
        self.linear1 = Linear(1024, 64)
        self.linear2 = Linear(64, 10)

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x) # torch.Size([64, 1024])
        x = self.linear1(x)
        x = self.linear2(x)
        return x


yn = YN()
optim = torch.optim.SGD(yn.parameters(), lr = 0.01)

for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = yn(imgs)
        ce_loss = loss_cross_entropy(outputs, targets)
        # Step 1, gradient back to zero
        optim.zero_grad()
        ce_loss.backward()
        optim.step()
        running_loss += ce_loss
    print(running_loss)
