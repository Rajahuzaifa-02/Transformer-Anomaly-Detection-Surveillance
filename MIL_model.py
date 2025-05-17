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