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