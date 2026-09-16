import os
import shutil
from sklearn.model_selection import train_test_split

# Configuration
input_images_path = './data/train/images'
input_labels_path = './data/train/labels'
output_base_dir = './dataset'

# Split ratios
train_ratio = 0.75
val_ratio = 0.15
test_ratio = 0.10

# Create output directories
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_base_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, split, 'labels'), exist_ok=True)

# Get all filenames (strip extension to match images with labels)
image_files = [f for f in os.listdir(input_images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
filenames = [os.path.splitext(f)[0] for f in image_files]

# First split: Separate train from the rest (val + test)
# 0.75 train, 0.25 remaining
train_files, remaining_files = train_test_split(filenames, train_size=train_ratio, random_state=42)

# Second split: Separate val and test from the remaining 25%
# Since 0.15 is 60% of 0.25 (0.15 / 0.25 = 0.6)
val_files, test_files = train_test_split(remaining_files, train_size=0.6, random_state=42)

def move_files(file_list, split_name):
    for name in file_list:
        # Find the original image extension
        ext = next(f.split('.')[-1] for f in image_files if f.startswith(name))

        # Paths
        src_img = os.path.join(input_images_path, f"{name}.{ext}")
        src_lbl = os.path.join(input_labels_path, f"{name}.txt")

        dest_img = os.path.join(output_base_dir, split_name, 'images', f"{name}.{ext}")
        dest_lbl = os.path.join(output_base_dir, split_name, 'labels', f"{name}.txt")

        # Copy files (shutil.copy2 preserves metadata)
        if os.path.exists(src_lbl):
            shutil.copy2(src_img, dest_img)
            shutil.copy2(src_lbl, dest_lbl)
        else:
            print(f"Warning: Label missing for {name}, skipping...")

# Execute the moves
print("Splitting data...")
move_files(train_files, 'train')
move_files(val_files, 'val')
move_files(test_files, 'test')
print(f"Done! Your dataset is ready in the '{output_base_dir}' folder.")


data_yaml = """
train: ./dataset/train
test: ./dataset/test
val: ./dataset/val

nc: 4

names:
  0: ball
  1: goalkeeper
  2: player
  3: referee"""

with open("./dataset/data.yaml", "w") as f:
  f.write(data_yaml)

print("data.yaml file created at './dataset/data.yaml'")