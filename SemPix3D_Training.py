import os
import torch
import random
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter




# -- DataLoader (On-the-fly Pair) --
class SRNShapesOnTheFlyPairDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None, test_split=2001):
        assert os.path.isdir(root_dir), f"Root dir not found: {root_dir}"
        self.root_dir = root_dir
        self.transform = transform or T.ToTensor()
        # Collect numeric instance names
        all_instances = [d for d in sorted(os.listdir(root_dir)) if d.isdigit()]
        self.instances = [inst for inst in all_instances if (int(inst) < test_split) == train]
        assert self.instances, "No instances found after split."

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        inst = self.instances[idx]
        rgb_dir = os.path.join(self.root_dir, inst, 'rgb')
        pose_dir = os.path.join(self.root_dir, inst, 'pose')
        v1, v2 = random.sample(range(250), 2)
        img1, p1 = self.load_view(rgb_dir, pose_dir, v1)
        img2, p2 = self.load_view(rgb_dir, pose_dir, v2)
        return {'image1': img1, 'pose1': p1, 'image2': img2, 'pose2': p2}

    def load_view(self, rgb_dir, pose_dir, vid):
        name = f"{vid:06d}"
        img_path = os.path.join(rgb_dir, name + '.png')
        pose_path = os.path.join(pose_dir, name + '.txt')
        img = Image.open(img_path).convert('RGB')
        img_t = self.transform(img)
        pose = torch.from_numpy(
            np.loadtxt(pose_path, dtype=np.float32).reshape(4,4)
        )
        return img_t, pose
    

def get_loader(root_dir, batch_size=8, num_workers=0, transform=None, train=True):
    ds = SRNShapesOnTheFlyPairDataset(root_dir, train=train, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True)



# -- PoseNet Definition --
class PoseNet(nn.Module):
    """PoseNet predicting a 4x4 transformation matrix (rotation + translation)."""
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Separate fully connected layers for rotation and translation
        self.fc_rot = nn.Linear(512, 6)   # Predict 6D rotation (better than 9D for stability)
        self.fc_trans = nn.Linear(512, 3) # Predict 3D translation

    def compute_rotation_matrix_from_6d(self, x):
        # Convert 6D representation to 3x3 rotation matrix using Gram-Schmidt
        x = x.view(-1, 6)
        a1 = F.normalize(x[:, 0:3], dim=1)
        a2 = x[:, 3:6]
        b2 = F.normalize(a2 - (a1 * a2).sum(dim=1, keepdim=True) * a1, dim=1)
        b3 = torch.cross(a1, b2, dim=1)
        R = torch.stack([a1, b2, b3], dim=-1)  # B x 3 x 3
        return R

    def forward(self, x):
        B = x.size(0)
        f = self.features(x)
        f = self.avgpool(f).view(B, -1)

        rot_6d = self.fc_rot(f)
        trans = self.fc_trans(f)

        R = self.compute_rotation_matrix_from_6d(rot_6d)  # B x 3 x 3
        T = trans.view(B, 3, 1)

        pose = torch.cat([R, T], dim=2)  # B x 3 x 4
        bottom = torch.tensor([0, 0, 0, 1], device=pose.device, dtype=pose.dtype).view(1, 1, 4).repeat(B, 1, 1)
        pose_mat = torch.cat([pose, bottom], dim=1)  # B x 4 x 4

        return pose_mat


import torch
import torch.nn as nn

class Pose2PoseNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder: image + source pose
        self.encoder_img = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),  # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),# 16 -> 8
            nn.ReLU()
        )

        self.pose1_fc = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

        self.encoder_fuse = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU()
        )

        # Decoder: encoded feat + target pose
        self.pose2_fc = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

        self.decoder_fuse = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8 → 16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 16 → 32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 32 → 64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),     # 64 → 128
            nn.Sigmoid()
        )

    def forward(self, img, pose1, pose2):
        B = img.size(0)
        pose1 = pose1.view(B, 16)
        pose2 = pose2.view(B, 16)

        # Encode image + source pose
        x_img = self.encoder_img(img)  # (B, 256, 8, 8)
        x_pose1 = self.pose1_fc(pose1).view(B, 256, 1, 1).expand(-1, -1, 8, 8)
        fused_enc = torch.cat([x_img, x_pose1], dim=1)  # (B, 512, 8, 8)
        feat = self.encoder_fuse(fused_enc)             # (B, 256, 8, 8)

        # Decode with target pose
        x_pose2 = self.pose2_fc(pose2).view(B, 256, 1, 1).expand(-1, -1, 8, 8)
        fused_dec = torch.cat([feat, x_pose2], dim=1)   # (B, 512, 8, 8)
        feat_dec = self.decoder_fuse(fused_dec)         # (B, 256, 8, 8)

        out_img = self.decoder(feat_dec)                # (B, 3, 128, 128)
        return out_img



H = W = 128
epochs = 10
root_dir = '../cars_train/cars_train'
batch_size = 8
num_workers = 0
checkpoint_dir = "Pose_Gen"
os.makedirs(checkpoint_dir, exist_ok=True)

# Transform
transform = transforms.Compose([
    transforms.Resize((H, W)),
    transforms.ToTensor(),
])

# DataLoader
train_loader = get_loader(root_dir, batch_size=batch_size, num_workers=num_workers, transform=transform, train=True)

# Models
posenet = PoseNet()
pose2posenet = Pose2PoseNet()

# Loss and optimizer
criterion_pose = nn.MSELoss()
criterion_img = nn.L1Loss()
optimizer_pose = optim.Adam(posenet.parameters(), lr=1e-4)
optimizer_pose2pose = optim.Adam(pose2posenet.parameters(), lr=1e-4)

# TensorBoard
writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, "runs"))

# Checkpoint loading
start_epoch = 0
checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    posenet.load_state_dict(checkpoint['posenet_state_dict'])
    pose2posenet.load_state_dict(checkpoint['pose2posenet_state_dict'])
    optimizer_pose.load_state_dict(checkpoint['optimizer_pose_state_dict'])
    optimizer_pose2pose.load_state_dict(checkpoint['optimizer_pose2pose_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"[INFO] Loaded checkpoint from epoch {start_epoch}")

# Training Loop
for epoch in range(start_epoch, epochs):
    posenet.train()
    pose2posenet.train()
    print("[",end="")
    for i, data in enumerate(train_loader):
        img1, pose1 = data['image1'], data['pose1']
        img2, pose2 = data['image2'], data['pose2']
        
        # PoseNet Forward
        pred_pose = posenet(img1)
        loss_pose = criterion_pose(pred_pose, pose1)

        optimizer_pose.zero_grad()
        loss_pose.backward()
        optimizer_pose.step()
        
        # Pose2PoseNet Forward
        pred_img = pose2posenet(img1, pose1, pose2)
        loss_img = criterion_img(pred_img, img2)

        optimizer_pose2pose.zero_grad()
        loss_img.backward()
        optimizer_pose2pose.step()
        if i % 5 == 0:
            print("=",end="")

    print("]")
    # Epoch-level logs
    writer.add_images("Samples/img1", img1, epoch)
    writer.add_images("Samples/img2", img2, epoch)
    writer.add_images("Samples/pred_img", pred_img, epoch)
    writer.add_text("Pose/pose1", str(pose1[0].view(-1).tolist()), epoch)
    writer.add_text("Pose/pose2", str(pose2[0].view(-1).tolist()), epoch)
    writer.add_text("Pose/pred_pose", str(pred_pose[0].view(-1).tolist()), epoch)
    writer.add_scalar("EpochLoss/pose", loss_pose.item(), epoch)
    writer.add_scalar("EpochLoss/image", loss_img.item(), epoch)

    # Save checkpoint every 2 epochs
    if (epoch + 1) % 2 == 0:
        torch.save({
            'epoch': epoch,
            'posenet_state_dict': posenet.state_dict(),
            'pose2posenet_state_dict': pose2posenet.state_dict(),
            'optimizer_pose_state_dict': optimizer_pose.state_dict(),
            'optimizer_pose2pose_state_dict': optimizer_pose2pose.state_dict(),
        }, checkpoint_path)
        print(f"[INFO] Checkpoint saved at epoch {epoch}")

writer.close()
