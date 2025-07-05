import torchvision
from torch.utils.tensorboard import SummaryWriter

# dataset_transform = torchvision.transforms.Compose([
#     torchvision.transforms.ToTensor()
# ])
dataset_transform = torchvision.transforms.ToTensor()


train_set = torchvision.datasets.CIFAR10(root = "./dataset", train = True, transform = dataset_transform, download = True)
test_set = torchvision.datasets.CIFAR10(root = "./dataset", train = False, transform = dataset_transform, download = True)

img, target = test_set[0] # img is tensor, target is int
# print(img.shape) # img.shape: torch.Size([3, 32, 32]),
# img.show()

# print(test_set[0])

writer = SummaryWriter("p5")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)
writer.close()