from torchvision import transforms
from PIL import Image
import cv2
from torch.utils.tensorboard import SummaryWriter

# Use transforms.toTensor to solve two problems
# 1. How to use transforms

# absolute path: /Users/wangyining/Desktop/github/Deep-Learning/pytorch_learn/hymenoptera_data/train/ants_image/0013035.jpg
# reletive path: hymenoptera_data/train/ants_image/0013035.jpg

img_path = "hymenoptera_data/train/ants_image/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(tensor_img.size()) #torch.Size([3, 512, 768])

# 2. Why do we use Tensor data type
cv_img = cv2.imread(img_path)
writer.add_image("Tensor_img", tensor_img)
writer.close()