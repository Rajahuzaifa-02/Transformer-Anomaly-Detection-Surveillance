import cv2
import os
import torch
from torchvision import transforms
from PIL import Image

def extract_frames(video_path, output_dir):
    """
    Extract frames from a video and save them in the specified output directory.

    :param video_path: Path to the video file.
    :param output_dir: Path to the output directory to save frames.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    success, frame = cap.read()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while success:
        frame_path = os.path.join(output_dir, f"frame_{frame_count:6d}.jpg")
        cv2.imwrite(frame_path, frame)
        success, frame = cap.read()
        frame_count += 1

    cap.release()

def process_folders_with_videos(base_folders, output_root_dir):
    """
    Process multiple folders containing videos and extract frames for each video.

    :param base_folders: List of folder paths containing video files.
    :param output_root_dir: Root directory where frames will be saved.
    """
    for folder in base_folders:
        if not os.path.exists(folder):
            print(f"Folder does not exist: {folder}")
            continue

        folder_name = os.path.basename(folder)  # Use the folder name for output structure
        output_folder_path = os.path.join(output_root_dir, folder_name)

        video_files = [f for f in sorted(os.listdir(folder)) if f.lower().endswith(('.mp4', '.avi', '.mkv'))]
        if not video_files:
            print(f"No video files found in folder: {folder}")
            continue

        for video_file in video_files:
            video_path = os.path.join(folder, video_file)
            video_name = os.path.splitext(video_file)[0]
            video_output_path = os.path.join(output_folder_path, video_name)

            print(f"Processing video: {video_path}")
            extract_frames(video_path, video_output_path)


folders_to_process = [
    'Anomaly_Videos',
    'Normal_Videos'
]

output_root_directory = 'New_model/Frames'

process_folders_with_videos(folders_to_process, output_root_directory)