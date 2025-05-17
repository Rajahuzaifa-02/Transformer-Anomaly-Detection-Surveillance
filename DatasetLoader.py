import os
import torch
from torch.utils.data import Dataset, DataLoader
import time

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize_segments(segments):
    """
    Normalize the segments to have zero mean and unit variance.

    Args:
        segments (torch.Tensor): The tensor containing segment data.
    
    Returns:
        torch.Tensor: Normalized segments.
    """
    norm = torch.norm(segments, p=2, dim=-1, keepdim=True)  # Compute L2 norm
    return segments / (norm + 1e-10)  # Normalize and prevent division by zero

def create_segments(features, window_size=5):
    """
    Creates overlapping segments from feature clips using a sliding window approach.
    
    Args:
        features (torch.Tensor): Tensor of shape (num_clips, feature_dim), where each row is a clip feature.
        window_size (int): Number of clips in each segment.

    Returns:
        torch.Tensor: Tensor of shape (num_segments, feature_dim), where each row is a segment feature.
    """
    num_clips, feature_dim = features.shape
    num_segments = num_clips - window_size + 1  # Sliding window count
    segments = []

    for i in range(0, num_segments, 5):
        segment = features[i : i + window_size].mean(dim=0)  # Compute mean of 5 clips
        segments.append(segment)

    return torch.stack(segments)

class VideoDataset(Dataset):
    def __init__(self, feature_dir, labels, transform=None, device='cuda'):
        self.feature_dir = feature_dir
        self.labels = labels
        self.transform = transform
        self.seg = create_segments
        self.device = device
        
        # Pre-load all segments and store them in memory
        self.segments_dict = {}
        self.labels_list = []
        
        for video_id, label in labels.items():
            video_segments = sorted([
                os.path.join(feature_dir, video_id, segment_file)
                for segment_file in os.listdir(os.path.join(feature_dir, video_id))
                if segment_file.endswith(".pt")  # Ensure only .pt files are included
            ])
           # print("Loading_", video_id)

            segments = [torch.load(segment_path, map_location=device) for segment_path in video_segments]
#            # Apply transformations if any
            if self.transform:
                segments = [self.transform(segment) for segment in segments]
            
            stacked_segments = torch.stack(segments)
            stacked_segments = stacked_segments.squeeze(1)
           # stacked_segments = self.seg(stacked_segments)
            self.segments_dict[video_id] = stacked_segments
            self.labels_list.append(label)
            
            # self.segments_list.append(torch.stack(segments))  # Stack segments into one tensor
            # self.labels_list.append(label)  # Corresponding label
        
        # Convert lists to tensors
        # print("Converting")
        # self.segments_tensor = torch.stack(self.segments_list).to(self.device)
        # self.labels_tensor = torch.tensor(self.labels_list, dtype=torch.float32).to(self.device)
        # print("Converted")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        video_id = list(self.segments_dict.keys())[idx]
        segments = self.segments_dict[video_id]
        label = self.labels_list[idx]

        # Return the segments and label for the video
        return video_id, segments, torch.tensor(label, dtype=torch.float32)


# # Create the dataset and dataloader for Normal
normal_video_dataset = VideoDataset(feature_dir='New_model/Normal', labels=normal)

normal_data_loader = DataLoader(normal_video_dataset, shuffle=True)

# # # Create the dataset and dataloader for Normal
anomaly_video_dataset = VideoDataset(feature_dir='New_model/Anomaly', labels=anomaly)

anomaly_data_loader = DataLoader(anomaly_video_dataset, shuffle=True)

# # # Create the dataset and dataloader for Normal
test_video_dataset = VideoDataset(feature_dir='New_model/Test', labels=test)

test_data_loader = DataLoader(test_video_dataset, shuffle=True)