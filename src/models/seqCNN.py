import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTM(nn.Module):
    '''
    Purpose: Process sequential frames (spatial + temporal features), 
    1. Feature Extraction with CNN
    2. Temporal Processing with LSTM:
    
    Args:
          inputs: sequence of tensors with shape (B, T, C, H, W) where
          B: Batch size.
          T: Number of frames per sequence processed.
          C: Number of channels (i.e. 3 for RGB).
          H, W: Height and width of each frame.

    '''
    def __init__(self, cnn_feature_extractor, hidden_size=128):
        super(CNNLSTM, self).__init__()
        self.cnn = cnn_feature_extractor 

        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, 
                            batch_first=True)  
    
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
       
        B, T, C, H, W = x.shape
        # B: Batch size.
        # T: Number of frames per sequence.
        # C: Number of channels (e.g., 3 for RGB).
        # H, W: Height and width of each frame.

        x = x.view(B*T, C, H, W)
        features = self.cnn(x)  # (B*T, 512)
                                
        features = features.view(B, T, -1)  # (B, T, 512)
                                            
        lstm_out, (h_n, c_n) = self.lstm(features)
                            
        predictions = self.fc(lstm_out)  #    final shape: (B, T, 1)
                                      
        return predictions  # (B, T, 1)


class ComplexCNNFeatureExtractor(nn.Module):

    '''
    Purpose: Extract spatial features from each image frame.
    '''

    def __init__(self):
        super(ComplexCNNFeatureExtractor, self).__init__()
    
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        
        # first convolution
        x = F.leaky_relu(self.bn1(self.conv1(x))) # first conv
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x)))) # second conv +  max pooling
        x = F.leaky_relu(self.bn3(self.conv3(x))) # third cov
        x = self.pool(F.leaky_relu(self.bn4(self.conv4(x)))) # forth conv +  max pooling
        x = F.leaky_relu(self.bn5(self.conv5(x))) # fifth conv
        
        x = self.global_avg_pool(x)    # (batch, 512, 1, 1)
                                        # globa average pooling
        x = x.view(x.size(0), -1)      # (batch, 512)
        return x

