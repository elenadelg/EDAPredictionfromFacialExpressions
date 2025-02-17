import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Code of 'Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks'
By Zitong Yu, 2019/05/05
If you use the code, please cite:
@inproceedings{yu2019remote,
    title={Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks},
    author={Yu, Zitong and Li, Xiaobai and Zhao, Guoying},
    booktitle= {British Machine Vision Conference (BMVC)},
    year = {2019}
}
Only for research purpose, and commercial use is not allowed.
MIT License
Copyright (c) 2019

Code from Github: https://github.com/Oichii/DeepPulse-pytorch/blob/master/PhysNet_SpaTemp.py (adjusted to handle BVP)
'''

class PhysNet_SpaTempBVP(nn.Module):
    """
    PhysNet with additional Spatio-Temporal layers.
    Using 3D Convs on input: (B, 3, T, H, W). We'll feed T=1.
    """
    def __init__(self, dropout_rate=0.5):
        super(PhysNet_SpaTempBVP, self).__init__()

        # Spatio-temporal blocks
        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]), 
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [1, 3, 3], stride=1, padding=[0, 1, 1]), 
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 32, [3, 1, 1], stride=1, padding=[1, 0, 0]), 
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(32, 32, [1, 3, 3], stride=1, padding=[0, 1, 1]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(32, 32, [3, 1, 1], stride=1, padding=[1, 0, 0]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(32, 64, [1, 3, 3], stride=1, padding=[0, 1, 1]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 1, 1], stride=1, padding=[1, 0, 0]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, [1, 3, 3], stride=1, padding=[0, 1, 1]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 1, 1], stride=1, padding=[1, 0, 0]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        self.ConvBlock10 = nn.Sequential(
            nn.Conv3d(64, 64, [1, 3, 3], stride=1, padding=[0, 1, 1]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock11 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 1, 1], stride=1, padding=[1, 0, 0]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock12 = nn.Sequential(
            nn.Conv3d(64, 64, [1, 3, 3], stride=1, padding=[0, 1, 1]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock13 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 1, 1], stride=1, padding=[1, 0, 0]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock14 = nn.Sequential(
            nn.Conv3d(64, 64, [1, 3, 3], stride=1, padding=[0, 1, 1]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )    
        self.ConvBlock15 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 1, 1], stride=1, padding=[1, 0, 0]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock16 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0) 
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_bvp = nn.Linear(1, 32)

        self.AvgpoolSpa1 = nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2))
        self.AvgpoolSpa2 = nn.AvgPool3d((1, 7, 7), stride=(1, 2, 2))
        self.MaxpoolSpa  = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))

        dummy_input = torch.zeros(1, 3, 1, 128, 128)  # T=1, H=128, W=128
        dummy_output = self.extract_spatiotemporal_features(dummy_input)
        flattened_size = dummy_output.view(dummy_output.size(0), -1).shape[1]

        self.fc_final = nn.Sequential(
            nn.Linear(flattened_size + 32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def extract_spatiotemporal_features(self, sequences):
        x = self.ConvBlock1(sequences)
        x = self.MaxpoolSpa(x)

        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.ConvBlock4(x)
        x = self.ConvBlock5(x)
        x = self.MaxpoolSpa(x)

        x = self.ConvBlock6(x)
        x = self.ConvBlock7(x)
        x = self.ConvBlock8(x)
        x = self.ConvBlock9(x)
        x = self.MaxpoolSpa(x)

        x = self.ConvBlock10(x)
        x = self.ConvBlock11(x)
        x = self.ConvBlock12(x)
        x = self.ConvBlock13(x)
        x = self.AvgpoolSpa1(x)

        x = self.ConvBlock14(x)
        x = self.ConvBlock15(x)
        x = self.AvgpoolSpa2(x)
        x = self.ConvBlock16(x)

       
        # We take x[:, :, -1, :, :] => last in time dimension
        return x[:, :, -1, :, :]  

    def forward(self, sequences, bvp_inputs):
        """
        sequences: shape (B, 3, 1, H, W) 
        bvp_inputs: shape (B,) => we map to (B,1) => fc_bvp => (B,32)
        """
        if sequences is None or bvp_inputs is None:
            raise ValueError("Both sequences (with T=1) and BVP inputs must be provided.")

    
        # PhysNet forward pass
        x = self.ConvBlock1(sequences)
        x = self.MaxpoolSpa(x)
    
        x = self.dropout(x)  

        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.ConvBlock4(x)
        x = self.ConvBlock5(x)
        x = self.MaxpoolSpa(x)
        x = self.dropout(x)  

        x = self.ConvBlock6(x)
        x = self.ConvBlock7(x)
        x = self.ConvBlock8(x)
        x = self.ConvBlock9(x)
        x = self.MaxpoolSpa(x)
        x = self.dropout(x)  

        x = self.ConvBlock10(x)
        x = self.ConvBlock11(x)
        x = self.ConvBlock12(x)
        x = self.ConvBlock13(x)
        x = self.AvgpoolSpa1(x)
        x = self.dropout(x)  

        x = self.ConvBlock14(x)
        x = self.ConvBlock15(x)
        x = self.AvgpoolSpa2(x)
        x = self.ConvBlock16(x)

        # Extract last time step => shape (B, C, H', W')
        last_time_step = x[:, :, -1, :, :]
        
        # Flatten
        flat_features = last_time_step.view(last_time_step.size(0), -1)
        
        # Process BVP => (B, 1) => fc_bvp => (B,32)
        bvp_features = self.fc_bvp(bvp_inputs.unsqueeze(-1))
                    
        # Concatenate
        combined_features = torch.cat((flat_features, bvp_features), dim=1)

        out = self.fc_final(combined_features)  # => (B,1)
        return out.squeeze(-1)
