from torch.utils.data import Dataset
from PIL import Image
import os

# img_path = "/Users/wangyining/Desktop/github/Deep-Learning/hymenoptera_data/train/ants/0013035.jpg"
# img = Image.open(img_path)
# print(img.size)

class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir # global variable for the following functions
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label
    
    def __len__(self):
        return len(self.img_path)


root_dir = "hymenoptera_data/train"
ants_label_dir = "ants"
bees_label_dir = "bees"

ants_dataset = MyData(root_dir, ants_label_dir) # 124
bees_dataset = MyData(root_dir, bees_label_dir) # 121

train_dataset = ants_dataset + bees_dataset # 245
print(len(train_dataset))  

img, label = ants_dataset[0]
img.show()
# print(ants_dataset.__getitem__(0))