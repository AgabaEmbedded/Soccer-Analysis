import zipfile
import pandas as pd
import cv2
import shutil
import numpy as np
import os
import configparser


def extract_zip(zip_path, extract_to):
    """
    Extracts a ZIP file to the specified directory.

    :param zip_path: Path to the .zip file
    :param extract_to: Directory where files will be extracted
    """
    try:
        # Validate that the file exists
        if not os.path.isfile(zip_path):
            print(f"Error: File '{zip_path}' does not exist.")
            return

        # Ensure the extraction directory exists
        os.makedirs(extract_to, exist_ok=True)

        # Open and extract the ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            print(f"Successfully extracted '{zip_path}' to '{extract_to}'")

    except zipfile.BadZipFile:
        print(f"Error: '{zip_path}' is not a valid ZIP file.")
    except PermissionError:
        print("Error: Permission denied while accessing files.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")




def delete_folder(folder_path):
  try:
      if os.path.exists(folder_path):
          shutil.rmtree(folder_path)  # Recursively delete folder and contents
          print(f"Folder '{folder_path}' deleted successfully.")
      else:
          print(f"Folder '{folder_path}' does not exist.")
  except Exception as e:
      print(f"Error deleting folder: {e}")



def letterbox_image(image, target_size=(640, 640), color=(114, 114, 114)):
    ih, iw = image.shape[:2]
    tw, th = target_size

    # Calculate the scaling factor while preserving aspect ratio
    scale = min(tw / iw, th / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    # Resize the image to the new proportional dimensions
    resized_image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # Create a solid canvas of the target size filled with the padding color
    canvas = np.full((th, tw, 3), color, dtype=np.uint8)

    # Compute coordinates to paste the resized image right in the center
    top = (th - nh) // 2
    left = (tw - nw) // 2

    canvas[top:top+nh, left:left+nw] = resized_image

    # Return the canvas, the scale factor, and the padding values (needed to update bounding boxes)
    return canvas, scale, (left, top)



def get_class_mapping(sequence_path):
    """
    Parses gameinfo.ini to map individual track IDs to our specific labels:
    0: player (including team left/right, goalkeeper)
    1: ball
    None: Ignore (referees, staff, background elements)
    """
    ini_path = os.path.join(sequence_path, "gameinfo.ini")
    track_map = {}

    if not os.path.exists(ini_path):
        return track_map

    config = configparser.ConfigParser()
    config.read(ini_path)

    if "Sequence" in config:
        for key, value in config["Sequence"].items():

            if key.startswith("trackletid_"):
                try:
                    track_id = int(key.split("_")[1])
                    role = value.lower()

                    if "ball" in role:
                        track_map[track_id] = 0
                    elif "goalkeeper" in role:
                        track_map[track_id] = 1
                    elif "player" in role:
                        track_map[track_id] = 2
                    elif "referee" in role:
                        track_map[track_id] = 3
                    else:
                        track_map[track_id] = None # Referees & staff get dropped
                except ValueError:
                    continue
    return track_map

def convert_to_yolo_format(input_dir, output_dir):
    # Reset work folders safely
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    image_dir = os.path.join(output_dir, "images")
    label_dir = os.path.join(output_dir, "labels")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # 1. Image Resolution Parameters
    orig_w, orig_h = 1920, 1080
    target_w, target_h = 1920, 1088

    # 2. Dynamic Letterbox Scale & Padding Math for 1920x1080 -> 640x640
    scale = min(target_w / orig_w, target_h / orig_h) # 640 / 1920 = ~0.3333
    nw = int(orig_w * scale)                          # 640
    nh = int(orig_h * scale)                          # 360
    pad_x = (target_w - nw) // 2                      # 0 px padding on sides
    pad_y = (target_h - nh) // 2                      # 140 px padding on top/bottom

    columns_name = {0: "frame ID", 1: "track ID", 2: "top x", 3: "top y", 4: "width", 5: "height"}

    for video in os.listdir(input_dir):
        video_path = os.path.join(input_dir, video)
        if not os.path.isdir(video_path):
            continue

        gt_path = os.path.join(video_path, "gt/gt.txt")
        if not os.path.exists(gt_path):
            continue

        # Load identity mapping from gameinfo.ini
        track_to_class_map = get_class_mapping(video_path)

        df = pd.read_csv(gt_path, sep=",", header=None)
        df.drop(columns=[6, 7, 8, 9], inplace=True)
        df.rename(columns=columns_name, inplace=True)

        # Process annotations
        for index, row in df.iterrows():
            if row["frame ID"] % 5 != 0:
                continue
            track_id = int(row["track ID"])

            # Filter tracks using the .ini dictionary mappings
            class_id = track_to_class_map.get(track_id, None)
            if class_id is None:
                continue  # Skip referees, staff, or unassigned tracking rows

            frame_id = int(row["frame ID"])
            top_x = float(row["top x"])
            top_y = float(row["top y"])
            width = float(row["width"])
            height = float(row["height"])

            # Compute pixel centers on original 1920x1080 coordinate landscape
            orig_cx = top_x + (width / 2.0)
            orig_cy = top_y + (height / 2.0)

            # Map coordinates onto the new 640x640 canvas (Applying scale and padding shifts)
            new_cx = (orig_cx * scale) + pad_x
            new_cy = (orig_cy * scale) + pad_y
            new_w = width * scale
            new_h = height * scale

            # Normalize coordinates strictly by 640 for YOLO constraints
            yolo_cx = new_cx / target_w
            yolo_cy = new_cy / target_h
            yolo_w = new_w / target_w
            yolo_h = new_h / target_h

            output_file = os.path.join(label_dir, f"{video}_{frame_id}.txt")
            yolo_line = f"{class_id} {yolo_cx:.6f} {yolo_cy:.6f} {yolo_w:.6f} {yolo_h:.6f}\n"

            with open(output_file, "a") as f:
                f.write(yolo_line)

        # Process and resize the image directories
        img1_dir = os.path.join(video_path, "img1")
        if os.path.exists(img1_dir):
            for frame in os.listdir(img1_dir):
                im_dir = os.path.join(img1_dir, frame)
                frame_num = int(frame[:-4])
                if frame_num % 5 != 0:
                    continue
                im_out_dir = os.path.join(image_dir, f"{video}_{frame_num}.jpg")

                img = cv2.imread(im_dir)
                if img is not None:
                    # letterbox_image must contain matching dimensions/logic
                    resized_img, _, _ = letterbox_image(img, (target_w, target_h))
                    cv2.imwrite(im_out_dir, resized_img)



zip_file_path = "./SoccerNet/tracking/train.zip"
output_directory = "train"

extract_zip(zip_file_path, output_directory)

columns_name= {0: "frame ID",
1: "track ID",
2: "top x",
3: "top y",
4: "width",
5: "height"
}
convert_to_yolo_format("./train/train", "./data/train")
print(len(os.listdir("./data/train/images")))