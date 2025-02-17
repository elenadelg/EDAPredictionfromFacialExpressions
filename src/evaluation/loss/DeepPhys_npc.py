import torch
import torch.nn as nn

class NegativePearsonCorrelation(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(NegativePearsonCorrelation, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        """
        Computes Negative Pearson Correlation for 4D input and 2D target tensors
        """
        try:
            if y_pred.dim() != y_true.dim():
                y_pred = y_pred.view(y_pred.size(0), -1)  # Flatten to (batch_size, features)
                y_pred = y_pred.mean(dim=1, keepdim=True)  # Average to (batch_size, 1)

            y_pred = y_pred.float()
            y_true = y_true.float()
            
            y_pred = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min() + self.epsilon) 
            pred_mean = y_pred.mean(dim=0) 
            true_mean = y_true.mean(dim=0)
            pred_diff = y_pred - pred_mean
            true_diff = y_true - true_mean
            
            # Compute correlation
            numerator = (pred_diff * true_diff).sum(dim=0)
            denominator = torch.sqrt(
                (pred_diff ** 2).sum(dim=0) * (true_diff ** 2).sum(dim=0) + self.epsilon
            )
            correlation = numerator / denominator
            correlation = torch.clamp(correlation, min=-1.0 + self.epsilon, max=1.0 - self.epsilon)
            
            return -correlation.mean()

        except Exception as e:
            print(f"Error in NPC loss: {str(e)}")
            raise e

