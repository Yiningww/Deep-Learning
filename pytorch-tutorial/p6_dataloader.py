import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  


# Test Dataset, length = 10000
test_data = torchvision.datasets.CIFAR10(root = "./dataset", train = False, download = True, transform = torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset = test_data, batch_size = 64, shuffle = False, num_workers = 0, drop_last = False)

# 1st img and target of test_data
img, target = test_data[0]
print(img.shape) # torch.Size([3, 32, 32])
print(target) # 3
writer = SummaryWriter("dataloader")
step = 0
for epoch in range(2):
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape) # torch.Size([batch_size, 3, 32, 32])
        # print(targets) # tensor([5, 0, 6, 2])
        writer.add_images(f"Epoch {epoch}", imgs, step)
        step += 1

