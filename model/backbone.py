import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm1d(nn.Module):
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x):  # x: [B, C, T]
        x = x.transpose(1, 2)      # [B, T, C]
        x = self.ln(x)
        return x.transpose(1, 2)   # [B, C, T]

class RMSNorm1d(nn.Module):
    def __init__(self, num_channels, eps=1e-8, affine=True):
        super().__init__()
        self.norm = nn.RMSNorm(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x):  # x: [B, C, T]
        x = x.transpose(1, 2)    # [B, T, C]
        x = self.norm(x)         # normalize over C
        return x.transpose(1, 2) # [B, C, T]

class ExtractEncoder(nn.Module):
    def __init__(self, in_chan):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_chan, 512, kernel_size=8, stride=5, padding=0, bias=False),
            #nn.BatchNorm1d(512),
            #LayerNorm1d(512),
            #RMSNorm1d(512),
            nn.GroupNorm(num_groups=256, num_channels=512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=8, stride=5, padding=0, bias=False),
            #nn.BatchNorm1d(512),
            LayerNorm1d(512),
            #RMSNorm1d(512),
            #nn.GroupNorm(num_groups=512, num_channels=512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=3, padding=0, bias=False),
            #nn.BatchNorm1d(512),
            LayerNorm1d(512),
            #RMSNorm1d(512),
            #nn.GroupNorm(num_groups=512, num_channels=512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=0, bias=False),
            #nn.BatchNorm1d(512),
            LayerNorm1d(512),
            #RMSNorm1d(512),
            #nn.GroupNorm(num_groups=512, num_channels=512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=0, bias=False),
            #nn.BatchNorm1d(512),
            LayerNorm1d(512),
            #RMSNorm1d(512),
            #nn.GroupNorm(num_groups=512, num_channels=512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.encoder(x)
    
class cpc_backbone(nn.Module):
    """
    Stable-Pretraining compatible CDCK2 backbone
    Only outputs time-series representations z_t (no CPC loss).
    """
    def __init__(self):
        super().__init__()
        self.encoder = ExtractEncoder(in_chan=1)
        self.gru = nn.GRU(512, 256, num_layers=1, bidirectional=False, batch_first=True)

    def forward(self, x):
        """
        Input: x [B, 1, T]
        Output: c_t (context features) [B, T', 256], z_t (encoded features) [B, T', 512]
        """
        z = self.encoder(x).transpose(1, 2)  # [B, T', 512]
        output, _ = self.gru(z)  # [B, T', 256]
        return output, z
        
