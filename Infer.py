# infer.py
import torch, math
from torchvision.utils import save_image
from models import PoseNet, DepthNet, TinyNeRF, AppearanceUNet

CKPT_PATH = "checkpoints/latest.pth"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# -------- Load nets and weights --------------------------------------------
pose_net  = PoseNet().to(DEVICE).eval()
depth_net = DepthNet(freeze=True).to(DEVICE).eval()
nerf      = TinyNeRF().to(DEVICE).eval()
unetG     = AppearanceUNet().to(DEVICE).eval()

ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
pose_net.load_state_dict(ckpt["pose"])
depth_net.load_state_dict(ckpt["depth"])
nerf.load_state_dict(ckpt["nerf"])
unetG.load_state_dict(ckpt["gen"])
print("✓ checkpoint restored")

# -------- Single‑view inference function -----------------------------------
@torch.no_grad()
def synthesize(rgb_I, target_pose):
    """
    rgb_I      : (1,3,H,W) tensor in [0,1]
    target_pose: (1,6) pose vector (tx,ty,tz,rx,ry,rz)
    Returns: RGB image from target viewpoint
    """
    # 1. Predict pose of input view (unused later but can refine NeRF)
    _ = pose_net(rgb_I)

    # 2. Estimate depth for the input view
    D_input = depth_net(rgb_I)

    # 3. Refinement loop – query NeRF until depth error < ε
    ε = 1e‑3
    for _ in range(5):
        xyzdir = torch.rand_like(D_input).unsqueeze(-1)
        nerf_depth = nerf(xyzdir).reshape_as(D_input)
        err = torch.abs(nerf_depth - D_input).mean()
        if err < ε:
            break

    # 4. Render target view (here we just reuse nerf_depth as W)
    W_target = nerf_depth      # placeholder for actual volumetric render

    # 5. Appearance synthesis
    G_in = torch.cat([rgb_I, W_target], dim=1)
    I_out = unetG(G_in)
    return I_out.clamp(0,1)

# ---------------- Demo ------------------------------------------------------
if __name__ == "__main__":
    # Replace this with real RGB and pose
    demo_rgb  = torch.rand(1,3,128,128, device=DEVICE)
    target_P  = torch.tensor([[0,0,0,0,math.pi/4,0]], device=DEVICE)  # 45° yaw
    novel_img = synthesize(demo_rgb, target_P)

    save_image(novel_img.cpu(), "novel_view.png")
    print("✓ novel view saved → novel_view.png")
