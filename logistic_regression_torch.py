import torch
import torch.nn as nn
import torch.optim as optim
from keras.datasets import mnist
import numpy as np
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.2)
train_x = train_x.reshape((train_x.shape[0],-1)) # reshape((rows,cols)), reshape((60000, -1))中-1是将“其他”维度展平为一维：(60000, 28, 28) -> (60000, 784)
val_x = val_x.reshape((val_x.shape[0],-1))
test_x = test_x.reshape((test_x.shape[0],-1))

class LogisticRegressionModel(nn.Module):
    def __init__(self, n_features, n_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(n_features, n_classes)  # 注意这里是n_classes

    def forward(self, x):
        # Softmax activation; no need to apply softmax here since we will use nn.CrossEntropyLoss
        return self.linear(x)

class FNN(nn.Module):
    def __init__(self, n_features, n_classes):
        super(FNN, self).__init__()
        self.layer1 = nn.Linear(n_features, 64)
        self.active1 = nn.ReLU()
        self.layer2 = nn.Linear(64, 32)
        self.active2 = nn.ReLU()
        self.linear = nn.Linear(32, n_classes)

    def forward(self, x):
        x = self.linear(self.active2(self.layer2(self.active1(self.layer1(x)))))
        return x



#x_train_tensor: (48000, 784)
x_train_tensor = torch.from_numpy(train_x).float()  # 转换为float类型的tensor
y_train_tensor = torch.from_numpy(train_y).long()   # 标签转换为long类型的tensor
x_val_tensor = torch.from_numpy(val_x).float()
y_val_tensor = torch.from_numpy(val_y).long()
x_test_tensor = torch.from_numpy(test_x).float()
y_test_tensor = torch.from_numpy(test_y).long()
n_features = train_x.shape[1]
print(n_features)
n_classes = 10  # 有10个类别
model = FNN(n_features, n_classes)
criterion = nn.CrossEntropyLoss()  # 这个损失函数内部处理了softmax
optimizer = optim.SGD(model.parameters(), lr=0.0001)
num_epochs = 50
print(x_train_tensor.shape)  # 输出 (样本数, 784)
print(y_train_tensor.shape)  # 输出 (样本数, ) 或 (样本数, 1)

# 创建Tensor数据集
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

# 创建DataLoader
batch_size = 512
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle = True)

train_loss, val_loss = [], []

for epoch in range(num_epochs):
    model.train()
    train_loss_of_each_epoch = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs) #forward
        loss = criterion(outputs, labels) 
        train_loss_of_each_epoch += loss.item()
        loss.backward() #backward
        optimizer.step() #update
    train_loss.append(train_loss_of_each_epoch/len(train_loader))
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}')
    model.eval()
    val_loss_of_each_epoch = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss_of_each_epoch += loss.item()
    val_loss.append(val_loss_of_each_epoch/len(val_loader))

plt.plot(train_loss, label = "train loss")
plt.plot(val_loss, label = "val loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


model.eval()
with torch.no_grad():
    outputs = model(x_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    total = y_test_tensor.size(0)
    correct = (predicted == y_test_tensor).sum().item()
    print(f'Accuracy: {correct / total:.4f}') #0.9077 when batch size = 256, epoch = 50
