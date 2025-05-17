import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import os
from timesformer_pytorch import TimeSformer
from torch import nn

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# processor = AutoImageProcessor.from_pretrained("facebook/timesformer-hr-finetuned-k600")
model = TimesformerModel.from_pretrained("facebook/timesformer-hr-finetuned-k600")
model.to(device)

# Function to create segments
def create_segments(frame_dir, segment_length=16, stride=16):
    frames = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir)])
    segments = []

    for start in range(0, len(frames) - segment_length + 1, stride):
        segment = frames[start:start + segment_length]
        segments.append(segment)

    return segments

# Extract features from a segment tensor
def extract_features(segment_tensor, model):
    # Add batch dimension to segment tensor
    # segment_tensor = segment_tensor.unsqueeze(0).to(device)  # Shape: [1, 16, 3, 224, 224]
    with torch.no_grad():
        features = model(**segment_tensor)  # Output: [1, feature_dim]
        cls_token = features.last_hidden_state[:, 0, :]  # Extract [CLS] token representation
        print("features",features.last_hidden_state.shape)
        print("cls", cls_token.shape)
    return cls_token.cpu()  # Move features back to CPU for saving

# Process a single category
def process_category(category_dir):
    for video_dir_name in sorted(os.listdir(category_dir)):  # Iterate through video folders
        # Ensure the full output path follows the category -> video structure
        output_dir = os.path.join("New_model/New_Features", category_dir.split("/")[-1], video_dir_name)
        
        if os.path.exists(output_dir) and any(os.listdir(output_dir)):
            print(f"Skipping {video_dir_name} as features already exist.")
            continue
            
        os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
        
        video_dir = os.path.join(category_dir, video_dir_name)
        if os.path.isdir(video_dir):  # Make sure it's a directory
            segments = create_segments(video_dir)
            video_id = video_dir_name  # Use video folder name as ID
            for i, segment in enumerate(segments):
                images = [Image.open(frame_path).convert("RGB") for frame_path in segment]
                segment_tensor = processor(images=images, return_tensors="pt").to(device)
                print(segment_tensor['pixel_values'].shape)
                #inputs['pixel_values'].shape
                features = extract_features(segment_tensor, model)

                # Save features with video ID and segment index
                torch.save(features, os.path.join(output_dir, f"{video_id}_segment_{i:6d}.pt"))
                print(f"Saved features for {video_id} segment {i:6d}")

# Process all categories in the dataset
def process_all_categories(root_dir):
    for category_name in os.listdir(root_dir):
    
        category_dir = os.path.join(root_dir, category_name)
        if os.path.isdir(category_dir):  # Make sure it's a directory
            print(f"Processing category: {category_name}")
          
            process_category(category_dir)

# Main execution
if __name__ == "__main__":
    frames_root_dir = "New_model/Frames"  # Your root directory containing categories
    process_all_categories(frames_root_dir)
    print("Feature extraction complete.")