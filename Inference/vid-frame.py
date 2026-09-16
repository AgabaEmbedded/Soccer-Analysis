import cv2
import os
import glob

def frames_to_video(images_dir, output_video_path, fps=25):
    # Grab all jpeg images and sort them numerically so the video plays in order
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))

    if not image_files:
        print(f"No images found in {images_dir}")
        return

    # Read the first image to dynamically extract widescreen dimensions
    first_frame = cv2.imread(image_files[0])
    height, width, layers = first_frame.shape

    # Initialize the video writer with the MP4V codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"Stitching {len(image_files)} frames into {output_video_path}...")
    for image_file in image_files:
        frame = cv2.imread(image_file)
        video.write(frame)

    video.release()
    print("Video compilation complete!")

# Usage: Point to one of your downloaded sequence image directories
images_input = "./test/test/SNMOT-116/img1"
video_output = "test_video_116.mp4"

frames_to_video(images_input, video_output, fps=25)