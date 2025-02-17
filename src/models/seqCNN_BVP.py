import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexCNNFeatureExtractor(nn.Module):
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
        """
        x shape: (batch_size, 3, H, W)
        returns a feature vector of shape (batch_size, 512)
        """
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))
        
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.pool(F.leaky_relu(self.bn4(self.conv4(x))))
        
        x = F.leaky_relu(self.bn5(self.conv5(x)))
        x = self.global_avg_pool(x)  # (batch_size, 512, 1, 1)
        x = x.view(x.size(0), -1)    # (batch_size, 512)
        return x


class CNNLSTMWithBVP(nn.Module):
    def __init__(self, feature_extractor, hidden_size, bvp_embed_size=32):
        super(CNNLSTMWithBVP, self).__init__()
        self.cnn = feature_extractor  # ComplexCNNFeatureExtractor

        # Small embedding layer to transform BVP from 1D -> bvp_embed_size
        self.bvp_embedding = nn.Linear(1, bvp_embed_size)

        # LSTM input size is CNN output (512) + BVP embed size
        self.lstm = nn.LSTM(input_size=512 + bvp_embed_size, 
                            hidden_size=hidden_size, 
                            batch_first=True)
        
        # Predict EDA from the LSTM’s hidden state
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, bvp):
        """
        x shape:   (B, T, 3, H, W)   # image sequences
        bvp shape: (B, T)           # BVP signal per frame
        returns:   (B, T, 1)        # EDA predictions per frame
        """
        B, T, C, H, W = x.shape

        x = x.view(B*T, C, H, W)       # (B*T, 3, H, W)
        
        features = self.cnn(x)        # (B*T, 512)
        
        features = features.view(B, T, -1)  # (B, T, 512)
        
        bvp_unsq = bvp.unsqueeze(-1).float()    # (B, T, 1)
        bvp_embed = F.relu(self.bvp_embedding(bvp_unsq))  # (B, T, bvp_embed_size)

        combined_features = torch.cat([features, bvp_embed], dim=-1)  
        
        outputs, _ = self.lstm(combined_features)  # (B, T, hidden_size)

        predictions = self.fc(outputs)  # (B, T, 1)

        return predictions
