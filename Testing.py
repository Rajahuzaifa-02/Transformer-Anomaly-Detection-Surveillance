import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import os
from transformers import AutoImageProcessor, TimesformerModel
from torch import nn
import time
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoImageProcessor.from_pretrained("facebook/timesformer-hr-finetuned-k600")
model = TimesformerModel.from_pretrained("facebook/timesformer-hr-finetuned-k600")
model.to(device)



# Function to extract frames from a video file
def extract_frames_from_video(video_path, segment_length=16):
    cap = cv2.VideoCapture(video_path)  # Open video file
    frames = []
    while True:
        ret, frame = cap.read()  # Read frame
        if not ret:  # If no more frames, break
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        frames.append(frame)

        # If enough frames for a segment, yield the segment
        if len(frames) == segment_length:
            yield frames
            frames = []  # Reset frame buffer for the next segment

    cap.release()

# Extract features from a segment tensor
def extract_features(segment_tensor, model):
    # Add batch dimension to segment tensor
    # segment_tensor = segment_tensor.unsqueeze(0).to(device)  # Shape: [1, 16, 3, 224, 224]
    with torch.no_grad():
        features = model(**segment_tensor)  # Output: [1, feature_dim]
        cls_token = features.last_hidden_state[:, 0, :]  # Extract [CLS] token representation
        # print("features",features.last_hidden_state.shape)
        # print("cls", cls_token.shape)
    return cls_token.cpu()  # Move features back to CPU for saving

# Load the trained anomaly detection model
trained_model = AnomalyDetectionModel().to(device)
trained_model.load_state_dict(torch.load("New_model/New_Model_3/model4000.pth"))
trained_model.eval()

# Real-time processing of videos
def process_video_real_time(video_path):
    for segment in extract_frames_from_video(video_path):
       # segment_tensor = preprocess_segment(segment)  # Preprocess segment
        #segment_tensor = segment.unsqueeze(0)  # Add batch dimension
                        #images = [Image.open(frame_path).convert("RGB") for frame_path in segment]
        segment_tensor = processor(images=segment, return_tensors="pt").to(device)
        with torch.no_grad():
            # Extract features
            features = extract_features(segment_tensor, model)
          #  features = features.cpu()  # Move features to CPU if needed
            features = features.to(device)
            # Make predictions using the trained model
            prediction = trained_model(features)
            prediction = prediction.cpu().numpy()  # Convert to NumPy array
            
            print(f"Prediction for segment: {prediction}")
        # You can save or process predictions as needed

# Main execution
if __name__ == "__main__":
    video_dir = "Arson"  # Directory containing video files
    for video_file in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video_file)
        if video_file.endswith(('.mp4', '.avi', '.mov')):  # Check for valid video file extensions
            print(f"Processing video: {video_file}")
            start_time = time.time()

            process_video_real_time(video_path)
            end_time = time.time()
            total_time = end_time - start_time
            
            print(f"Total process time: {total_time:.4f} seconds")

    print("Real-time video processing complete.")
    