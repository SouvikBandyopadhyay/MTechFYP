# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------- Pose Generator ----------
class PoseNet(nn.Module):
    """
    Very small CNN+MLP that regresses a 6‑DoF pose vector
    (3 × translation, 3 × Euler rotation) from a single RGB view.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, 6)

    def forward(self, x: Tensor) -> Tensor:               # (B,3,H,W) → (B,6)
        x = self.encoder(x).flatten(1)
        return self.head(x)

# ---------- Depth Estimator ----------
class DepthNet(nn.Module):
    """
    Simple UNet‑like monocular depth estimator.
    Pre‑train it or freeze it as required.
    """
    def __init__(self, freeze: bool = False):
        super().__init__()
        self.down1 = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU())
        self.down2 = nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU())
        self.mid  = nn.Sequential(nn.Conv2d(64,128,3,1,1), nn.ReLU())
        self.up1  = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.up2  = nn.ConvTranspose2d(64, 1,   4, 2, 1)
        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:               # (B,3,H,W) → (B,1,H,W)
        x1 = self.down1(x); x2 = self.down2(x1)
        x  = self.mid(x2)
        x  = F.relu(self.up1(x))
        return self.up2(x)

# ---------- Tiny NeRF Core ----------
class TinyNeRF(nn.Module):
    """
    Minimal MLP for density σ and depth regression given (x,y,z,dir).
    In training we supply depth GT from the DepthNet; during inference
    we loop queries until error < threshold.
    """
    def __init__(self, depth_feat=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, depth_feat), nn.ReLU(),
            nn.Linear(depth_feat, depth_feat), nn.ReLU(),
            nn.Linear(depth_feat, 1),  # predicted depth
        )

    def forward(self, xyz_dir: Tensor) -> Tensor:         # [...,4] → [...,1]
        return self.mlp(xyz_dir)

# ---------- Appearance Generator (UNet) ----------
class AppearanceUNet(nn.Module):
    """
    Inputs: concatenated (source RGB I1, NeRF render W)
            E.g. channels = 3 + 1 = 4
    Outputs: RGB image from target view.
    """
    def __init__(self, in_ch=4, base=64):
        super().__init__()
        self.down1 = nn.Sequential(nn.Conv2d(in_ch, base,3,1,1), nn.ReLU())
        self.down2 = nn.Sequential(nn.Conv2d(base, base*2,3,2,1), nn.ReLU())
        self.mid   = nn.Sequential(nn.Conv2d(base*2, base*4,3,1,1), nn.ReLU())
        self.up1   = nn.ConvTranspose2d(base*4, base*2,4,2,1)
        self.out   = nn.Conv2d(base*2, 3, 3, 1, 1)

    def forward(self, x: Tensor) -> Tensor:               # (B,4,H,W) → (B,3,H,W)
        x1  = self.down1(x); x2 = self.down2(x1)
        xmid= self.mid(x2)
        xup = F.relu(self.up1(xmid))
        return torch.sigmoid(self.out(xup))

# ---------- PatchGAN Discriminator ----------
class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base,4,2,1), nn.LeakyReLU(0.2),
            nn.Conv2d(base, base*2,4,2,1), nn.BatchNorm2d(base*2), nn.LeakyReLU(0.2),
            nn.Conv2d(base*2, 1, 4,1,1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
