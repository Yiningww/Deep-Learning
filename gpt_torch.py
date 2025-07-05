import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

# 定义超参数
batch_size = 64
learning_rate = 0.01
num_epochs = 10

# 加载 MNIST 数据集
train_dataset = MNIST(root='./data', train=True, download=True)
test_dataset = MNIST(root='./data', train=False, download=True)

# 数据归一化函数
def normalize(tensor):
    return (tensor - 0.5) / 0.5

# 划分训练集和验证集
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# 定义数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# 定义 Logistic Regression 模型
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.view(-1, 28*28)  # 将图像展平
        return self.linear(x)

# 初始化模型、损失函数和优化器
input_size = 28 * 28  # MNIST 图像是 28x28
num_classes = 10  # 数字分类：0-9
model = LogisticRegressionModel(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 训练模型，并记录损失
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for images, labels in train_loader:
        # 数据归一化
        images = normalize(images.float())
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
    
    # 计算并记录训练损失
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # 在验证集上评估
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = normalize(images.float())
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
    
    # 计算并记录验证损失
    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

# 绘制训练和验证损失图
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

