import torch
import torch.nn as nn

class NegativePearsonCorrelation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        """
        Computes the Negative Pearson Correlation between predictions and targets,
        treating the entire batch as one set of (prediction,target) pairs. 
        Both y_pred and y_true are expected to be 1D or 2D with a single value per sample.
        """
        y_pred = y_pred.view(-1).float()
        y_true = y_true.view(-1).float()

        pred_mean = torch.mean(y_pred)
        true_mean = torch.mean(y_true)
        pred_dev = y_pred - pred_mean
        true_dev = y_true - true_mean

        # Covariance and variances
        covariance = torch.sum(pred_dev * true_dev)
        pred_var   = torch.sum(pred_dev ** 2)
        true_var   = torch.sum(true_dev ** 2)

        # Pearson correlation (scalar)
        pearson_corr = covariance / (torch.sqrt(pred_var) * torch.sqrt(true_var) + 1e-8)

        return -pearson_corr # negative correlation as the "loss"


