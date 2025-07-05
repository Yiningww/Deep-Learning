import os

root_dir = "hymenoptera_data/train"
target_dir = "ants_image"
label = target_dir.split("_")[0]
img_path = os.listdir(os.path.join(root_dir, target_dir))
out_dir = os.path.join(root_dir, "ants_label")
for img in img_path:
    file_name = img.split(".")[0]
    with open(os.path.join(out_dir, f"{file_name}.txt"), "w") as f:
        f.write(label)