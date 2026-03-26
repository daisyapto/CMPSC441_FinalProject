import os
import random
import shutil

# Original dataset
source_dir = "../Data"

# New directories
train_dir = "../Data/train"
test_dir = "../Data/test"

classes = ["Healthy", "Brain_Tumor"]

# Create folders
for cls in classes:
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

# Split ratio
split_ratio = 0.8  # 80% train, 20% test

for cls in classes:
    cls_folder = os.path.join(source_dir, cls)
    images = os.listdir(cls_folder)

    random.shuffle(images)

    split_index = int(len(images) * split_ratio)
    train_images = images[:split_index]
    test_images = images[split_index:]

    for img in train_images:
        src = os.path.join(cls_folder, img)
        dst = os.path.join(train_dir, cls, img)
        shutil.copy(src, dst)

    for img in test_images:
        src = os.path.join(cls_folder, img)
        dst = os.path.join(test_dir, cls, img)
        shutil.copy(src, dst)

print("Dataset split complete.")
