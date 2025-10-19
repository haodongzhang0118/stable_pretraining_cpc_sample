import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ExtractEncoder(nn.Module):
    def __init__(self, in_chan):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_chan, 512, kernel_size=8, stride=5, padding=2, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=8, stride=5, padding=2, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=3, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(512),
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
        