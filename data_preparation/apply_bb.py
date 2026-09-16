import os
import cv2
import random
import shutil
import numpy as np
from PIL import Image

train_image_dir = r"./data/train/images"
train_label_dir = r"./data/train/labels"
output_dir = r"./with_boxes"

os.makedirs(output_dir, exist_ok=True)
shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

# Get list of image files
image_files = [f for f in os.listdir(train_image_dir) if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")]

# Randomly select a few images
num_samples = min(20, len(image_files))  # Select up to 5 images
selected_images = random.sample(image_files, num_samples)
print(f"{len(selected_images)} images found")
# Function to draw bounding boxes
def draw_bounding_boxes(image_path, label_path, output_path):
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    # Read label file
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            data = line.strip().split()
            if len(data) != 5:
                continue

            class_id, x_center, y_center, bbox_width, bbox_height = map(float, data)

            # Convert YOLO format to pixel coordinates
            x_center, y_center = int(x_center * w), int(y_center * h)
            bbox_width, bbox_height = int(bbox_width * w), int(bbox_height * h)

            xmin = int(x_center - bbox_width / 2)
            ymin = int(y_center - bbox_height / 2)
            xmax = int(x_center + bbox_width / 2)
            ymax = int(y_center + bbox_height / 2)

            class_ids = {0: "b",
                          1: "g",
                          2: "p",
                          3: "r"}

            # Draw bounding box
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
            cv2.putText(image, class_ids[int(class_id)], (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


    #cv2.imshow("image", image)
    # Save the image with bounding boxes
    cv2.imwrite(output_path, image)

    # Process selected images
for img_file in selected_images:

    image_path = os.path.join(train_image_dir, img_file)


    label_file = img_file.replace(".png", ".txt").replace(".jpg", ".txt").replace(".jpeg", ".txt")
    #label_file = label_file.replace("image", "label")

    label_path = os.path.join(train_label_dir, label_file)

    output_path = os.path.join(output_dir, img_file)

    draw_bounding_boxes(image_path, label_path, output_path)
    #on print(img_file.replace(".png", ""))

print(f"Processed {num_samples} images with bounding boxes.")
