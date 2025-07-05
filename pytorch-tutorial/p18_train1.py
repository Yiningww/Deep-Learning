import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
from model import *
from torch.utils.tensorboard import SummaryWriter


train_data = torchvision.datasets.CIFAR10(root = "dataset", train = True,
                                       transform = torchvision.transforms.ToTensor(),
                                       download = True)


test_data = torchvision.datasets.CIFAR10(root = "dataset", train = False,
                                       transform = torchvision.transforms.ToTensor(),
                                       download = True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"training data length is: {train_data_size}")
print(f"testing data length is: {test_data_size}")

train_dataloader = DataLoader(train_data, batch_size = 64)
test_dataloader = DataLoader(test_data, batch_size = 64)

# Create NN
yn = YN()

loss_fn = nn.CrossEntropyLoss()

# Oprimizer
learning_rate = 1e-2
optimizer = torch.optim.SGD(yn.parameters(), lr = learning_rate)

# Set NN Parameters
total_train_step = 0 # record training times

total_test_step = 0 # record testing times

epoch = 1

# Add TensorBoard
writer = SummaryWriter("p18")



for i in range(epoch):
    print(f"-----------Start Training of Epoch {i + 1}----------")
    loss_each_epoch = 0
    yn.train()
    for data in train_dataloader:
        imgs, targets = data
        outputs = yn(imgs)
        loss = loss_fn(outputs, targets)
        # Optimizer optimize model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_each_epoch += loss
        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f"training no.{total_train_step}, Loss: {loss.item()}") # tensor(5) -> 5
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    # print(loss_each_epoch)

    # Test Part
    yn.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = yn(imgs)
            loss = loss_fn(outputs, targets)
            total_test_step += 1
            # print(loss)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print(f"test loss each epoch: {total_test_loss}")
    print(f"test accuracy: {total_accuracy/test_data_size}")
    writer.add_scalar("test_loss", loss.item(), total_test_step)
    writer.add_scalar("test_acc", total_accuracy, total_test_step)

    total_test_step += 1

    torch.save(yn, "yn_{}.pth".format(i))
    # torch.save(yn.state_dict(), f"yn_{i}.pth")
    print("model saved")

writer.close()


    
