from pathlib import Path
import random
import shutil
import math


# ============================================================
# CONFIGURATION
# ============================================================

# Folder containing ALL your images
IMAGES_DIR = Path(
    r"C:\Users\Agaba_Embedded4\Desktop\Soccer-Tracking\soccer-torso-number-6\train\images"
)

# Folder containing the annotation TXT files
LABELS_DIR = Path(
    r"C:\Users\Agaba_Embedded4\Desktop\Soccer-Tracking\soccer-torso-number-6\train\labels"
)

# Folder to create
OUTPUT_DIR = Path(
    r"prepared_dataset"
)

# Null images as a percentage of annotated images
NULL_RATIO = 0.15

# Dataset split
TRAIN_RATIO = 0.75
VAL_RATIO = 0.15
TEST_RATIO = 0.10

# Reproducibility
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# VALIDATE CONFIGURATION
# ============================================================

if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-6:
    raise ValueError(
        "TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0"
    )

random.seed(RANDOM_SEED)


# ============================================================
# FIND ALL IMAGES
# ============================================================

all_images = [
    image
    for image in IMAGES_DIR.iterdir()
    if image.is_file()
    and image.suffix.lower() in IMAGE_EXTENSIONS
]

print(f"Total images found: {len(all_images)}")


# ============================================================
# SEPARATE ANNOTATED AND NULL IMAGES
# ============================================================

annotated = []
null_images = []

for image in all_images:

    label = LABELS_DIR / f"{image.stem}.txt"

    if label.exists() and label.stat().st_size > 0:
        annotated.append((image, label))
    else:
        null_images.append(image)


print(f"Annotated images: {len(annotated)}")
print(f"Null images available: {len(null_images)}")


# ============================================================
# SELECT NULL IMAGES
# ============================================================

num_null = math.floor(
    len(annotated) * NULL_RATIO
)

if len(null_images) < num_null:
    raise ValueError(
        f"Not enough null images.\n"
        f"Need: {num_null}\n"
        f"Available: {len(null_images)}"
    )

selected_null = random.sample(
    null_images,
    num_null
)

print(f"Null images selected: {len(selected_null)}")


# ============================================================
# CREATE COMPLETE DATASET
# ============================================================

dataset = []

# Annotated images
for image, label in annotated:
    dataset.append((image, label))

# Null images
for image in selected_null:
    dataset.append((image, None))


# Randomize everything before splitting
random.shuffle(dataset)

print(f"Total dataset images: {len(dataset)}")


# ============================================================
# SPLIT DATASET
# ============================================================

total = len(dataset)

train_count = int(total * TRAIN_RATIO)
val_count = int(total * VAL_RATIO)

train_data = dataset[:train_count]

val_data = dataset[
    train_count:
    train_count + val_count
]

test_data = dataset[
    train_count + val_count:
]


print("\nDataset split:")
print(f"Train:      {len(train_data)}")
print(f"Validation: {len(val_data)}")
print(f"Test:       {len(test_data)}")


# ============================================================
# CREATE DIRECTORY STRUCTURE
# ============================================================

for split in ["train", "validation", "test"]:

    (OUTPUT_DIR / split / "images").mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT_DIR / split / "labels").mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# COPY DATA
# ============================================================

def copy_split(data, split):

    image_dir = (
        OUTPUT_DIR /
        split /
        "images"
    )

    label_dir = (
        OUTPUT_DIR /
        split /
        "labels"
    )

    annotated_count = 0
    null_count = 0

    for image, label in data:

        # Copy image
        shutil.copy2(
            image,
            image_dir / image.name
        )

        # Copy annotation if it exists
        if label is not None:

            shutil.copy2(
                label,
                label_dir / label.name
            )

            annotated_count += 1

        else:

            # Null image intentionally has
            # no TXT annotation.
            null_count += 1

    print(
        f"{split}: "
        f"{len(data)} images | "
        f"{annotated_count} annotated | "
        f"{null_count} null"
    )


# ============================================================
# BUILD EACH SPLIT
# ============================================================

copy_split(train_data, "train")
copy_split(val_data, "validation")
copy_split(test_data, "test")


# ============================================================
# CREATE data.yaml
# ============================================================

yaml_content = f"""path: {OUTPUT_DIR.as_posix()}

train: train/images
val: validation/images
test: test/images

names:
  0: "0"
  1: "1"
  2: "2"
  3: "3"
  4: "4"
  5: "5"
  6: "6"
  7: "7"
  8: "8"
  9: "9"
"""

yaml_path = OUTPUT_DIR / "data.yaml"

with open(
    yaml_path,
    "w",
    encoding="utf-8"
) as f:
    f.write(yaml_content)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DATASET PREPARATION COMPLETE")
print("=" * 60)

print(f"Annotated images: {len(annotated)}")
print(f"Null images:      {len(selected_null)}")
print(f"Total images:     {len(dataset)}")

print("\nFinal split:")
print(
    f"Train:      {len(train_data)} "
    f"({TRAIN_RATIO * 100:.0f}%)"
)

print(
    f"Validation: {len(val_data)} "
    f"({VAL_RATIO * 100:.0f}%)"
)

print(
    f"Test:       {len(test_data)} "
    f"({TEST_RATIO * 100:.0f}%)"
)

print("\nDataset structure:")
print(f"{OUTPUT_DIR}/")
print("├── train/")
print("│   ├── images/")
print("│   └── labels/")
print("├── validation/")
print("│   ├── images/")
print("│   └── labels/")
print("├── test/")
print("│   ├── images/")
print("│   └── labels/")
print("└── data.yaml")

print(f"\ndata.yaml: {yaml_path}")