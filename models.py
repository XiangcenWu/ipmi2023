import torch
import torch.nn as nn


class SegmentationNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 3, (3, 3, 3), 1, 1)

    def forward(self, x):
        return self.conv(x)


class SelectionNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(1, 1, 16, 16),
            nn.Flatten()
        )
        self.qkv = nn.Linear(64, 3*64, False)
        self.out = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.sm = nn.Softmax(dim=-1)
    def forward(self, x):
        x = self.feature_extractor(x)
        qkv = self.qkv(x).view(4, 3, 64).permute(1, 0, 2)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = q @ k.transpose(0, 1)
        a = self.sm(a)
        return self.out(a@v)
