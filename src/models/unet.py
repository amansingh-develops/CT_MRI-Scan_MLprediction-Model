"""
U-Net Model for Liver & Tumor Segmentation
Architecture follows FasNet paper (Nature Scientific Reports, 2025)
DOI: 10.1038/s41598-025-98427-9

Input:  [B, 1, 256, 256]  — grayscale CT slice
Output: [B, 2, 256, 256]  — 2-channel mask (liver + tumor)

CRITICAL RULES (paper-backed):
  - NO sigmoid inside forward() — applied only in loss/inference
  - ConvTranspose2d for upsampling (NOT bilinear interpolation)
  - bias=False in Conv2d when followed by BatchNorm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=2, feat=(64, 128, 256, 512)):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        ch = in_ch
        for f in feat:
            self.downs.append(DoubleConv(ch, f))
            ch = f
        self.bottleneck = DoubleConv(feat[-1], feat[-1] * 2)

        rev = list(reversed(feat))
        up_in = feat[-1] * 2
        for f in rev:
            self.ups.append(nn.ConvTranspose2d(up_in, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(f * 2, f))
            up_in = f

        self.head = nn.Conv2d(feat[0], out_ch, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i // 2]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = self.ups[i + 1](torch.cat([skip, x], dim=1))
        return self.head(x)


if __name__ == "__main__":
    # Sanity check (per FasNet paper spec)
    print("Testing U-Net (FasNet-aligned architecture)...")
    model = UNet(in_channels=1, out_channels=2)
    x = torch.randn(4, 1, 256, 256)
    out = model(x)
    print(f"Input shape:  {x.shape}")    # must be [4, 1, 256, 256]
    print(f"Output shape: {out.shape}")  # must be [4, 2, 256, 256]
    assert out.shape == (4, 2, 256, 256), f"Shape mismatch! Got {out.shape}"
    print("U-Net sanity check PASSED")

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
