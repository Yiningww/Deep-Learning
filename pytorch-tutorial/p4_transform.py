from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
# img = Image.open("nnmm.jpg").convert("RGB")
image_path = "hymenoptera_data/train/ants_image/0013035.jpg"
img = Image.open(image_path) # (768, 512)


# ToTensor : PIL -> ToTensor -> Tensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img) # Equal to: img_tensor = transforms.ToTensor()(img)
print(img_tensor.size()) # torch.Size([3, 512, 768])
writer.add_image("Tensor_img", img_tensor)

# Normalize: Ensure there are 3 channels, Tensor -> Normalize -> Tensor
print(img_tensor[0][0][0]) # tensor(0.3137)
trans_norm = transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5], inplace = True)
img_norm = trans_norm(img_tensor)
print(img_tensor[0][0][0]) # tensor(-0.3725) --> 0.3137 * 2 - 1
writer.add_image("Norm_img", img_norm)

# Resize: img PIL -> Resize -> img_resize PIL -> ToTensor -> img_resize_tensor
print(img.size) # (768, 512)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img) # PIL with size (512, 512)
print(img_resize)
img_resize_tensor = transforms.ToTensor()(img_resize) # (3, 512, 512)
writer.add_image("Resize_img", img_resize_tensor)

# Compose - resize
trans_resize_2 = transforms.Resize(512)
# PIL -> PIL, PIL -> Tensor
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize2 = trans_compose(img)
writer.add_image("Resize2_img", img_resize2)

# RandomCrop
# trans_randomcrop = transforms.RandomCrop(512) 
trans_randomcrop = transforms.RandomCrop((500, 1000)) 

trans_compose2 = transforms.Compose([trans_randomcrop, trans_totensor])
for i in range(10):
    img_crop = trans_compose2(img)
    writer.add_image("RandomCrop_imgHW", img_crop, i)

writer.close()



