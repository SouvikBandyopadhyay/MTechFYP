# train.py 
import os, itertools, torch, tqdm, random
from torch.utils.data import DataLoader
from models import PoseNet, DepthNet, TinyNeRF, AppearanceUNet, PatchDiscriminator

# ─────────────── Hyper‑params & paths ────────────────────────────────────────
BATCH_SIZE        =  4
N_EPOCHS          = 50
LR_START, LR_END  = 1e-3, 4e-4                        # dynamic schedule
CKPT_PATH         = "checkpoints/latest.pth"
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
seg_loss_weight   = 5.0                              # weight for segment similarity

# ─────────────── Dataset placeholder ───────────────────────────────────
class MultiViewSet(torch.utils.data.Dataset):
    """
    Replace with real multiview loader that also returns a segmentation mask
    for every RGB frame (shape H×W, integer labels).
    """
    def __len__(self): return 400
    def __getitem__(self, idx):
        H,W = 128,128
        imgs   = torch.rand(3,3,H,W)               # I1,I2,I3
        poses  = torch.rand(3,6)                   # P1,P2,P3
        segmap = torch.randint(0,8,(H,W))          # 8 random segments
        return imgs, poses, segmap

# ─────────────── Segment similarity loss ─────────────────────────────────────
def segment_similarity(i_fake, i_real, seg):
    """
    i_fake,i_real : (B,3,H,W)  – assumed 0‑1 range
    seg           : (H,W) or (B,H,W)  – integer segment IDs (0…K‑1)
    Returns scalar L1 difference of segment‑wise mean colours.
    """
    if seg.ndim == 2:
        seg = seg.unsqueeze(0).expand(i_fake.size(0),-1,-1)  # B,H,W
    B,_,H,W = i_fake.shape
    loss = 0.0
    K    = seg.max().item() + 1
    for k in range(K):
        mask = (seg == k).float().view(B,1,H,W)            # (B,1,H,W)
        if mask.sum() < 1: continue
        μ_fake = (i_fake * mask).sum(dim=[2,3]) / mask.sum(dim=[2,3])
        μ_real = (i_real * mask).sum(dim=[2,3]) / mask.sum(dim=[2,3])
        loss  += torch.abs(μ_fake - μ_real).mean()
    return loss / K

# ─────────────── Linear LR scheduler helper ─────────────────────────────────
def linear_lr(step, total_steps):
    """Returns multiplicative factor: 1.0 → LR_START, 0.0 → LR_END."""
    t = step / total_steps
    return (1 - t) * LR_START + t * LR_END

# ─────────────── Main training loop ─────────────────────────────────────────
def main():
    ds      = MultiViewSet()
    loader  = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    pose_net  = PoseNet().to(DEVICE)
    depth_net = DepthNet(freeze=False).to(DEVICE)
    nerf      = TinyNeRF().to(DEVICE)
    unetG     = AppearanceUNet().to(DEVICE)
    disc      = PatchDiscriminator().to(DEVICE)

    # ── Optimisers -----------------------------------------------------------
    optim_G = torch.optim.Adam(
        itertools.chain(pose_net.parameters(),
                        depth_net.parameters(),
                        nerf.parameters(),
                        unetG.parameters()),
        lr=LR_START, betas=(0.5,0.999))
    optim_D = torch.optim.Adam(disc.parameters(), lr=LR_START, betas=(0.5,0.999))

    # Scheduler implemented manually each step
    total_steps = len(loader) * N_EPOCHS
    global_step = 0

    bce = torch.nn.BCEWithLogitsLoss()
    l1  = torch.nn.L1Loss()

    for epoch in range(1, N_EPOCHS+1):
        pbar = tqdm.tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        for imgs, poses, segmap in pbar:
            imgs, poses = imgs.to(DEVICE), poses.to(DEVICE)
            segmap      = segmap.to(DEVICE)

            I1, I2 = imgs[:,0], imgs[:,1]
            P1, P2 = poses[:,0], poses[:,1]

            # ── Pose loss ───────────────────────────────────────────────────
            P_pred    = pose_net(I1)
            loss_pose = bce(torch.sigmoid(P_pred), torch.sigmoid(P1))

            # ── Depth & NeRF losses ────────────────────────────────────────
            D_pred_I1 = depth_net(I1).detach()
            xyzdir    = torch.rand_like(D_pred_I1).unsqueeze(-1)  
            nerf_depth= nerf(xyzdir).reshape_as(D_pred_I1)
            loss_nerf = l1(nerf_depth, D_pred_I1)

            # ── Appearance generation ─────────────────────────────────────
            W         = nerf_depth.detach()                    
            G_in      = torch.cat([I1, W], dim=1)
            I_fake    = unetG(G_in)

            # ── Discriminator update ──────────────────────────────────────
            disc_real = disc(I2)
            disc_fake = disc(I_fake.detach())
            valid, fake = torch.ones_like(disc_real), torch.zeros_like(disc_fake)
            loss_D     = 0.5*(bce(disc_real, valid) + bce(disc_fake, fake))

            optim_D.zero_grad(); loss_D.backward(); optim_D.step()

            # ── Generator‑side losses (adv + L1 + segment similarity) ─────
            disc_fake_for_G = disc(I_fake)
            loss_G_adv = bce(disc_fake_for_G, valid)
            loss_G_l1  = l1(I_fake, I2)
            loss_seg   = segment_similarity(I_fake, I2, segmap) * seg_loss_weight

            loss_G = loss_pose + loss_nerf + loss_G_adv + 100*loss_G_l1 + loss_seg

            optim_G.zero_grad(); loss_G.backward(); optim_G.step()

            # ── Dynamic LR step ───────────────────────────────────────────
            global_step += 1
            lr_now = linear_lr(global_step, total_steps)
            for pg in optim_G.param_groups: pg["lr"] = lr_now
            for pg in optim_D.param_groups: pg["lr"] = lr_now
            pbar.set_postfix(lr=f"{lr_now:.6f}")

        # ── Save (overwrite) checkpoint each epoch ────────────────────────
        os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
        torch.save({
            "pose": pose_net.state_dict(),
            "depth": depth_net.state_dict(),
            "nerf": nerf.state_dict(),
            "gen" : unetG.state_dict(),
            "disc": disc.state_dict()
        }, CKPT_PATH)
        print(f"✓ Epoch {epoch} complete – checkpoint updated")

if __name__ == "__main__":
    main()
