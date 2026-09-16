import os


dataset_path = r"soccer-torso-number-6"
labeled_dataset_path = r"soccer-torso-number-5"
for dir in os.listdir(dataset_path+r"/images"):
    print(f"Checking {dir} in labeled dataset...")
    if dir in os.listdir(labeled_dataset_path+r"/images"):
        os.remove(os.path.join(dataset_path,r"images", dir))
        os.remove(os.path.join(dataset_path, r"labels", dir.replace(".jpg", ".txt")))
        with open(os.path.join(labeled_dataset_path, r"images", dir), 'rb') as f:
            label = f.read()
        if len(label) == 0:
            os.remove(os.path.join(labeled_dataset_path, r"images", dir))
            os.remove(os.path.join(labeled_dataset_path, r"labels", dir.replace(".jpg", ".txt")))