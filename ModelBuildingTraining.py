import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#from timesformer_pytorch import TimeSformer  # Ensure you have TimeSformer installed or implemented
import json

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


def train_one_epoch(epoch,model, normal_data_loader, anomaly_data_loader, optimizer, device):
    model.train()
    optimizer.zero_grad()

   # epoch_loss = 0
    sub_sum = []
    smoothness_penalty = []
    anomaly_max = []
    normal_max = []
    total_loss = 0
   
    
    file_path = "New_model/TTT/segment_scores_evolution.json"
    if not os.path.exists(file_path):  
        segment_scores_evolution = {
            "iteration": [],
            "Video_id_anomaly": [],
            "anomaly_segment_scores": [],
            "Video_id_normal": [],
            "normal_segment_scores": []
            }
    else:
        with open(file_path, "r") as f:
            segment_scores_evolution = json.load(f)

    loss = 0
    for (normal_batch_idx, (normal_video_id, normal_segments, normal_labels)), (anomaly_batch_idx, (anomaly_video_id, anomaly_segments, anomaly_labels)) in zip(enumerate(normal_data_loader), enumerate(anomaly_data_loader)):

        if (normal_batch_idx + 1)  % 20 == 0 and normal_batch_idx != 0:
            
            anomaly_tensor = anomaly_max
            normal_tensor = normal_max

            sub_sum = torch.tensor(sub_sum)
            smoothness_penalty = torch.tensor(smoothness_penalty)

            loss = custom_objective_function(normal_tensor, anomaly_tensor, sub_sum, smoothness_penalty, device)
            total_loss += loss
            
            print(f"loss of Batch: {normal_batch_idx + 1} is :{loss}")
            print("-------------------------------------------------")

            loss.backward()
            optimizer.step() 
            
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"Layer: {name}, Gradient Norm: {param.grad.norm().item()}")
            
            print(f"Processing Batch: {normal_batch_idx + 1}")
            
            anomaly_max = []
            normal_max = []
            sub_sum = []
            smoothness_penalty = []
            
            #model.train()
            optimizer.zero_grad()
            
            normal_features, nor_labels = normal_segments.to(device), normal_labels.to(device)
            anomaly_features, anom_labels = anomaly_segments.to(device), anomaly_labels.to(device)

            anomaly_scores = model(anomaly_features)
            normal_scores = model(normal_features)
            
            anomaly_scores = anomaly_scores.squeeze(dim=-1)  # Remove the last dimension to get shape [1, 267] 
            anomaly_max.append(torch.max(anomaly_scores, dim=1)[0]) # Max score for each video


            normal_scores = normal_scores.squeeze(dim=-1)  # Remove the last dimension to get shape [1, 267]
            normal_max.append(torch.max(normal_scores, dim=1)[0]) # Max score for each video

            sub_sum.append(torch.sum(anomaly_scores))
            
            diff = anomaly_scores[:, 1:] - anomaly_scores[:, :-1]
            smoothness_penalty.append(torch.sum(torch.square(diff)))
           # print(smoothness_penalty)

   

        else:
            print(f"Processing Batch: {normal_batch_idx + 1}")

            normal_features, nor_labels = normal_segments.to(device), normal_labels.to(device)
            anomaly_features, anom_labels = anomaly_segments.to(device), anomaly_labels.to(device)

            anomaly_scores = model(anomaly_features)
            normal_scores = model(normal_features)
            
            anomaly_scores = anomaly_scores.squeeze(dim=-1)  # Remove the last dimension to get shape [1, 267] 
            anomaly_max.append(torch.max(anomaly_scores, dim=1)[0]) # Max score for each video


            normal_scores = normal_scores.squeeze(dim=-1)  # Remove the last dimension to get shape [1, 267]
            normal_max.append(torch.max(normal_scores, dim=1)[0]) # Max score for each video

            sub_sum.append(torch.sum(anomaly_scores))
            
            diff = anomaly_scores[:, 1:] - anomaly_scores[:, :-1]
            smoothness_penalty.append(torch.sum(torch.square(diff)))


            if epoch % 200 == 0:

                if normal_batch_idx % 50 == 0:
                    # Store all segment scores for the selected video
                    selected_anomaly_scores = anomaly_scores[0].detach().cpu().numpy()  # Full scores for anomaly video
                    selected_normal_scores = normal_scores[0].detach().cpu().numpy()    # Full scores for normal video
                    
    
                    
                    # with open(file_path, "r") as f:
                    #     segment_scores_evolution = json.load(f)
    
                                    # Append to dictionary
                    segment_scores_evolution["iteration"].append(normal_batch_idx + 1)
                    segment_scores_evolution["Video_id_anomaly"].append(anomaly_video_id)
                    segment_scores_evolution["anomaly_segment_scores"].append(selected_anomaly_scores.tolist())
                    segment_scores_evolution["Video_id_normal"].append(normal_video_id)
                    segment_scores_evolution["normal_segment_scores"].append(selected_normal_scores.tolist())
    
                    
                    with open(file_path, "w") as f:
                        json.dump(segment_scores_evolution, f, indent=4)  # Indent for better readability



                    
                    print(f"Saved segment scores for batch {epoch}")
                    torch.save(model.state_dict(), f"New_model/TTT/model{epoch}.pth")
                    torch.save(optimizer.state_dict(), f"New_model/TTT/optimizer{epoch}.pth")
                    print(f"Saved model and optimizer state at epoch {epoch}")




    return total_loss

# Define your model, optimizer, and loss
model = AnomalyDetectionModel().cuda()
optimizer = optim.Adagrad(model.parameters(), lr=0.01)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)
file_path = "New_model/New_Model_4/training_log.txt"
if not os.path.exists(file_path):
    with open(file_path, 'w') as f:
        f.write("Iteration, Total Loss\n")


num_epochs = 4000

for epoch in range(1, num_epochs + 1):
    print(f"Epoch {epoch}/{num_epochs}")

    # Train
    train_loss = train_one_epoch(epoch,model, normal_data_loader,  anomaly_data_loader, optimizer, device)
    print(f"Full Training Loss: {train_loss:.4f}")
        # Append the new iteration and loss
    with open(file_path, 'a') as f:
        f.write(f"{epoch}, {train_loss:.6f}\n")

print("Training Complete!")
