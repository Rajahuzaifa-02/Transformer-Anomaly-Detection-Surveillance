import torch
import torch.nn as nn
import torch.nn.functional as F

class AnomalyDetectionModel(nn.Module):
    def __init__(self, feature_dim=768):
        super(AnomalyDetectionModel, self).__init__()
        
        # Adding an additional hidden layer with more neurons
        self.fc1 = nn.Linear(feature_dim, 32)   # Increased size of first layer
      #  self.fc2 = nn.Linear(256, 32)              # Output layer
        self.fc3 = nn.Linear(32, 1)              # Output layer
        
        self.dropout = nn.Dropout(0.6)  # Slightly reduced dropout rate for regularization
        #self.activation = nn.LeakyReLU(0.1)
    # self.batch_norm1 = nn.BatchNorm1d(128)   # Batch Normalization after first layer
        #self.batch_norm2 = nn.BatchNorm1d(32)    # Batch Normalization after second layer
        
    def forward(self, x):
        # First hidden layer with Batch Normalization and ReLU activation
        # Apply BatchNorm1d correctly: we need to permute so that the feature dimension (128) is in the second place
        #x = self.dropout(x)

        x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        # # Now apply batch normalization over the second dimension (feature dimension)
        # x = x.permute(0, 2, 1)  # [batch_size, feature_dim, sequence_length] -> [1, 128, 164]
        # x = self.batch_norm1(x)  # BatchNorm expects [batch_size, num_features, seq_length
        # x = x.permute(0, 2, 1)  # Restore to original shape [1, 164, 128]
        # x = self.dropout(x)
        
        # # # Second hidden layer with Batch Normalization and ReLU activation


        # x = x.permute(0, 2, 1)
        # x = self.batch_norm2(x)  # BatchNorm expects [batch_size, num_features, seq_length]
        # x = x.permute(0, 2, 1)  # Restore to original shape [1, 164, 64]

     #   x = self.ReLU(self.fc2(x))
       # x = self.dropout(x)
         # Third hidden layer
        # x = F.relu(self.fc3(x))
        x = self.dropout(x)
        
        # Output layer with Sigmoid activation for binary classification (anomaly detection)
        x = torch.sigmoid(self.fc3(x))
        
        return x


#def compute_mil_loss(positive_bag, negative_bag, lambda1=8e-5, lambda2=8e-5, lambda3=0.01):

def custom_objective_function(normal_scores_list, anomaly_scores_list, sub_sum, smoothness_penalty, device, lambda1=8e-5, lambda2=8e-5, lambda3=0.01):
    """
    Custom Objective function to calculate Ranking Loss, Temporal Smoothness, and Sparsity.

    Parameters:
        normal_scores_list: List of average top 10 normal scores for each batch.
        anomaly_scores_list: List of average top 10 anomaly scores for each batch.
        normal_labels: Labels for normal videos.
        anomaly_labels: Labels for anomalous videos.
        device: The device (GPU or CPU) to run the function on.
    
    Returns:
        loss: A scalar value representing the total loss.
    """
    # print(normal_scores_list)
    # print(anomaly_scores_list)
    normal_scores = torch.stack(normal_scores_list).to(device)
    anomaly_scores = torch.stack(anomaly_scores_list).to(device)
    # print(normal_scores)
    # print(anomaly_scores) 
    ranking_loss = torch.tensor(0.0, device=device)
    for normal_score in normal_scores:
        ranking_loss += torch.sum(F.relu(1 - anomaly_scores + normal_scores))
    ranking_loss /= len(normal_scores)

  #  ranking_loss = torch.mean(F.relu(1 - anomaly_scores + normal_scores))  # Ensures anomaly scores > normal scores

   
   # l2_reg = sum(param.norm(2) for param in model.parameters())
    print(ranking_loss)
    # 
    loss = ranking_loss + lambda1 * torch.sum(sub_sum) + lambda2 * torch.sum(smoothness_penalty) + 0.01 * (sum(param.norm(2) for param in model.parameters()))
    #+ lambda3 * l2_reg
    return loss

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
trained_model.load_state_dict(
    torch.load("New_model/New_Model_3/model4000.pth", map_location=torch.device('cpu'))
)
#trained_model.load_state_dict(torch.load("New_model/New_Model_3/model4000.pth"))
trained_model.eval()


# Real-time processing of videos
def process_video_real_time(video_path):
    
    pred = []
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
            pred.append(prediction)

            #  # If anomaly detected (threshold can be adjusted)
            # if prediction > 0.5:  # Assuming 0.5 is the threshold for anomaly detection
            #     alert_list.append(f"Anomaly detected at frame {len(alert_list) + 1}!")

            # Dynamically update the alert section
            # with alert_placeholder.container():
            #     st.subheader("Real-Time Alerts")
            #     for alert in alert_list:
            #         st.write(alert)

            #print(f"Prediction for segment: {prediction}")
            yield f"data: Prediction for segment: {prediction}\n\n"
   # return StreamingResponse(generate(), media_type="text/event-stream")
    #return pred
        # You can save or process predictions as needed

# # Main execution
# if __name__ == "__main__":
#     video_dir = "test"  # Directory containing video files
#     for video_file in os.listdir(video_dir):
#         video_path = os.path.join(video_dir, video_file)
#         if video_file.endswith(('.mp4', '.avi', '.mov')):  # Check for valid video file extensions
#             print(f"Processing video: {video_file}")
#             start_time = time.time()

#             process_video_real_time(video_path)
#             end_time = time.time()
#             total_time = end_time - start_time
            
#             print(f"Total process time: {total_time:.4f} seconds")

#     print("Real-time video processing complete.")


