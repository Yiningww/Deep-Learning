from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

# image_path = "hymenoptera_data/train/ants_image/0013035.jpg"
# image = Image.open(image_path)
# img = np.array(image) # (512, 768, 3)-->(H, W, C)
# print(img.shape)
writer = SummaryWriter("logs")
image_path = "hymenoptera_data/train/ants_image/0013035.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(img_array.shape)
writer.add_image("test", img_array, 1, dataformats="HWC")
# y = x
for i in range(100):
    writer.add_scalar("y = 2x", 3 * i, i)
writer.close()


# from torch.utils.tensorboard import SummaryWriter
# import numpy as np
# from PIL import Image

# writer = SummaryWriter("logs")
# image_path = "hymenoptera_data/train/ants_image/0013035.jpg"
# img_PIL = Image.open(image_path)
# img_array = np.array(img_PIL) # <class 'numpy.ndarray'>, size: (512, 768, 3)
# print(type(img_array))
# print(img_array.shape)

# writer.add_image("train", img_array, 1, dataformats='HWC')
# # y = 2x
# for i in range(100):
#     writer.add_scalar("y=2x", 3*i, i)

# writer.close()