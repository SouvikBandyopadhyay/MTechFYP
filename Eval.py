import argparse, math, random, itertools, warnings
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────────────────
# Metrics from torchmetrics
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore
# ────────────────────────────────────────────────────────────────────────────


# ---------------- Dataset ---------------------------------------------------
class FlatImageFolder(Dataset):
    """Reads *all* image files under a directory (recursively)."""
    IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

    def __init__(self, root: Path, transform=None):
        self.paths = [p for p in root.rglob("*") if p.suffix.lower() in self.IMG_EXT]
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found under {root}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img) if self.transform else img


# ---------------- Diversity -------------------------------------------------
@torch.no_grad()
def diversity(features: torch.Tensor, max_pairs: int = 50_000) -> float:
    """
    Average Euclidean distance between randomly‑picked feature pairs.
    `features` : (N, D) tensor on *CPU*.
    """
    n = features.size(0)
    if n < 2:
        return 0.0
    # Choose up to max_pairs random pairs without replacement
    idx = torch.combinations(torch.arange(n), r=2)
    if idx.size(0) > max_pairs:
        perm = torch.randperm(idx.size(0))[:max_pairs]
        idx = idx[perm]
    diff = features[idx[:, 0]] - features[idx[:, 1]]
    dist = torch.norm(diff, dim=1)          # L2
    return dist.mean().item()


# ---------------- Main ------------------------------------------------------
def evaluate(real_dir: Path, gen_dir: Path,
             batch: int = 64,
             device: str = "cpu",
             num_workers: int = 4):

    # Standard InceptionV3 preprocessing (torchmetrics normalises internally)
    tfm = transforms.Compose([
        transforms.Resize(299, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
    ])

    real_ds = FlatImageFolder(real_dir, tfm)
    gen_ds  = FlatImageFolder(gen_dir,  tfm)

    real_loader = DataLoader(real_ds, batch_size=batch, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    gen_loader  = DataLoader(gen_ds,  batch_size=batch, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    # ── Metrics initialisation ──────────────────────────────────────────────
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    kid = KernelInceptionDistance(subset_size=100, resize=False,
                                  normalize=True).to(device)
    isc = InceptionScore(split=10, normalize=True).to(device)

    # Containers to collect generated‑feature tensor for diversity
    gen_feats = []

    # ── Pass 1: real images (update FID/KID with real=True) ────────────────
    print("Embedding REAL images …")
    for imgs in tqdm(real_loader, unit="batch"):
        imgs = imgs.to(device)
        fid.update(imgs, real=True)
        kid.update(imgs, real=True)

    # ── Pass 2: generated images (update real=False, +IS, +div) ────────────
    print("Embedding GENERATED images …")
    for imgs in tqdm(gen_loader, unit="batch"):
        imgs = imgs.to(device)
        fid.update(imgs, real=False)
        kid.update(imgs, real=False)
        isc.update(imgs)

        # features for diversity (hook from fid metric)
        with torch.no_grad():
            feats = fid.inception(imgs)
        gen_feats.append(feats.cpu())

    # ── Compute final scores ───────────────────────────────────────────────
    fid_score = fid.compute().item()
    kid_mean, kid_std = kid.compute()
    kid_mean, kid_std = kid_mean.item(), kid_std.item()
    is_mean, is_std = isc.compute()
    is_mean, is_std = is_mean.item(), is_std.item()

    div_score = diversity(torch.cat(gen_feats, dim=0))

    print("\n────────────  Results  ────────────")
    print(f"FID : {fid_score:8.3f}")
    print(f"KID : {kid_mean*1000:8.3f} ± {kid_std*1000:.3f}  (×10‑3)")
    print(f"IS  : {is_mean:8.3f} ± {is_std:.3f}")
    print(f"Diversity (L2 on Inception feats): {div_score:8.3f}")
    print("───────────────────────────────────")


# ---------------- CLI -------------------------------------------------------
def _parse():
    p = argparse.ArgumentParser(description="Evaluate GAN outputs.")
    p.add_argument("--real_dir", required=True, type=Path,
                   help="Directory with reference/real images")
    p.add_argument("--gen_dir",  required=True, type=Path,
                   help="Directory with generated images")
    p.add_argument("--batch",    default=64,  type=int, help="Batch size")
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu",
                   choices=["cpu", "cuda"], help="Computation device")
    p.add_argument("--workers",  default=4, type=int,
                   help="Data‑loader worker processes")
    return p.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    args = _parse()
    evaluate(args.real_dir, args.gen_dir,
             batch=args.batch,
             device=args.device,
             num_workers=args.workers)