# core/losses/cdck2_loss.py
import math
import torch
import torch.nn.functional as F
from torch import nn

class cpc_loss(nn.Module):
    def __init__(self, tau=0.07):
        super().__init__()
        self.tau = tau

    def forward(self, preds, encode_samples):
        """
        preds: [timestep, B, D]
        encode_samples: [timestep, B, D]
        """
        timestep, B, D = preds.shape
        total_loss, total_correct = 0.0, 0

        for i in range(timestep):
            pred_raw = preds[i]
            enc_raw = encode_samples[i]

            scale = 1.0 / math.sqrt(D)
            logits = (enc_raw @ pred_raw.t()) * scale / self.tau
            labels = torch.arange(B, device=pred_raw.device)
            total_loss += F.cross_entropy(logits, labels, reduction="mean")

            pred_label = logits.argmax(dim=1)
            total_correct += (pred_label == labels).sum().item()

        return total_loss / timestep, float(total_correct) / (B * timestep)
