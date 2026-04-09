"""
DWDM Project — gan_generator.py
GAN (Generative Adversarial Network) for AI Image Generation
=====================================================================
Architecture: DCGAN (Deep Convolutional GAN)
  - Generator   : Noise vector → 64×64 RGB image
  - Discriminator: 64×64 RGB image → Real/Fake probability

Dataset path: C:\\Users\\shash\\Downloads\\DATASET\\DATASET\\cleaned_data\\splits

Usage:
    # Train the GAN
    python gan_generator.py --mode train

    # Generate new AI images using trained generator
    python gan_generator.py --mode generate --num_images 100

    # Generate + show grid of samples
    python gan_generator.py --mode generate --num_images 64 --show_grid

Requirements:
    pip install torch torchvision Pillow matplotlib numpy tqdm
"""

import os
import sys
import argparse
import time
import json
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────
DATASET_PATH  = r"C:\Users\shash\Downloads\DATASET\DATASET\cleaned_data\splits\train\ai"
MODELS_DIR    = r"C:\Users\shash\Downloads\DATASET\DATASET\models"
GAN_OUTPUT_DIR= r"C:\Users\shash\Downloads\DATASET\DATASET\gan_outputs"

# GAN Hyperparameters
IMG_SIZE      = 64       # output image size (64×64) — DCGAN standard
CHANNELS      = 3        # RGB
NOISE_DIM     = 100      # latent noise vector dimension
G_FEATURES    = 64       # generator feature map base size
D_FEATURES    = 64       # discriminator feature map base size
BATCH_SIZE    = 64
EPOCHS        = 100      # increase to 200+ for better quality
LR_G          = 2e-4     # generator learning rate
LR_D          = 2e-4     # discriminator learning rate
BETA1         = 0.5      # Adam beta1 (DCGAN recommendation)
BETA2         = 0.999
SAVE_INTERVAL = 10       # save checkpoint every N epochs
SAMPLE_INTERVAL= 5       # save sample images every N epochs
NUM_WORKERS   = 0        # 0 for Windows
# ─────────────────────────────────────────────────────────────────

os.makedirs(MODELS_DIR,     exist_ok=True)
os.makedirs(GAN_OUTPUT_DIR, exist_ok=True)
SAMPLES_DIR = os.path.join(GAN_OUTPUT_DIR, "training_samples")
GENERATED_DIR = os.path.join(GAN_OUTPUT_DIR, "generated_images")
os.makedirs(SAMPLES_DIR,   exist_ok=True)
os.makedirs(GENERATED_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  DWDM — DCGAN: AI Image Generator")
print("="*65)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    print("  [✓] PyTorch available:", torch.__version__)
except ImportError:
    print("  [ERROR] PyTorch not found.")
    print("  Install: pip install torch torchvision")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL = True
except ImportError:
    MPL = False

try:
    from tqdm import tqdm
    TQDM = True
except ImportError:
    TQDM = False


# ══════════════════════════════════════════════════════════════════
# DEVICE
# ══════════════════════════════════════════════════════════════════
def get_device():
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print(f"  [✓] GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        d = torch.device("mps")
        print("  [✓] Apple MPS GPU")
    else:
        d = torch.device("cpu")
        print("  [i] CPU mode (training will be slow — GPU recommended)")
    return d


# ══════════════════════════════════════════════════════════════════
# WEIGHT INITIALISATION  (DCGAN paper: mean=0, std=0.02)
# ══════════════════════════════════════════════════════════════════
def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ══════════════════════════════════════════════════════════════════
# GENERATOR ARCHITECTURE
# ══════════════════════════════════════════════════════════════════
class Generator(nn.Module):
    """
    DCGAN Generator:
    Input  : noise vector (NOISE_DIM × 1 × 1)
    Output : RGB image   (3 × 64 × 64)

    Architecture (4 transposed conv blocks):
      noise(100) → 4×4 → 8×8 → 16×16 → 32×32 → 64×64 (RGB)
    """
    def __init__(self, noise_dim=NOISE_DIM, g_feat=G_FEATURES, channels=CHANNELS):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            # Block 1: noise_dim × 1×1  →  g_feat*8 × 4×4
            self._block(noise_dim,   g_feat * 8, 4, 1, 0),  # 1→4

            # Block 2: g_feat*8 × 4×4  →  g_feat*4 × 8×8
            self._block(g_feat * 8, g_feat * 4, 4, 2, 1),  # 4→8

            # Block 3: g_feat*4 × 8×8  →  g_feat*2 × 16×16
            self._block(g_feat * 4, g_feat * 2, 4, 2, 1),  # 8→16

            # Block 4: g_feat*2 × 16×16 → g_feat × 32×32
            self._block(g_feat * 2, g_feat,     4, 2, 1),  # 16→32

            # Output: g_feat × 32×32 → channels × 64×64
            nn.ConvTranspose2d(g_feat, channels, 4, 2, 1, bias=False),  # 32→64
            nn.Tanh()   # output in range [-1, 1]
        )

    def _block(self, in_ch, out_ch, kernel, stride, pad):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
        )

    def forward(self, noise):
        return self.net(noise)


# ══════════════════════════════════════════════════════════════════
# DISCRIMINATOR ARCHITECTURE
# ══════════════════════════════════════════════════════════════════
class Discriminator(nn.Module):
    """
    DCGAN Discriminator:
    Input  : RGB image   (3 × 64 × 64)
    Output : scalar probability [0, 1]  (real=1, fake=0)

    Architecture (4 conv blocks):
      64×64 → 32×32 → 16×16 → 8×8 → 4×4 → 1×1 (scalar)
    """
    def __init__(self, d_feat=D_FEATURES, channels=CHANNELS):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            # Block 1: channels × 64×64 → d_feat × 32×32
            # No BatchNorm on first layer (DCGAN recommendation)
            nn.Conv2d(channels, d_feat, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 2: d_feat × 32×32 → d_feat*2 × 16×16
            self._block(d_feat,     d_feat * 2, 4, 2, 1),

            # Block 3: d_feat*2 × 16×16 → d_feat*4 × 8×8
            self._block(d_feat * 2, d_feat * 4, 4, 2, 1),

            # Block 4: d_feat*4 × 8×8 → d_feat*8 × 4×4
            self._block(d_feat * 4, d_feat * 8, 4, 2, 1),

            # Output: d_feat*8 × 4×4 → 1 × 1×1
            nn.Conv2d(d_feat * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # probability output
        )

    def _block(self, in_ch, out_ch, kernel, stride, pad):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, image):
        return self.net(image).view(-1, 1).squeeze(1)


# ══════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════
def get_dataloader(data_dir: str, img_size: int, batch_size: int):
    """
    Load images from folder.
    Normalise to [-1, 1] to match Generator Tanh output.
    """
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],   # mean per channel
                             [0.5, 0.5, 0.5]),   # std per channel
    ])

    # ImageFolder expects: data_dir/class_name/image.jpg
    # If data_dir is directly the image folder, wrap it
    path = Path(data_dir)
    if not any(p.is_dir() for p in path.iterdir()):
        # No subdirectories — use parent and filter to this folder
        # Create a temporary structure-compatible path
        parent = path.parent
        dataset = datasets.ImageFolder(str(parent), transform=tf)
        # Filter to only images from the target class
        class_idx = dataset.class_to_idx.get(path.name)
        if class_idx is not None:
            indices = [i for i, (_, label) in enumerate(dataset.samples) if label == class_idx]
            from torch.utils.data import Subset
            dataset = Subset(dataset, indices)
    else:
        dataset = datasets.ImageFolder(data_dir, transform=tf)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
        pin_memory=True,
    )
    print(f"  [✓] Dataset loaded: {len(dataset):,} images from {data_dir}")
    return loader


# ══════════════════════════════════════════════════════════════════
# SAVE SAMPLE GRID
# ══════════════════════════════════════════════════════════════════
def save_sample_grid(generator, fixed_noise, epoch, device, nrow=8):
    """Generate and save a grid of sample images."""
    generator.eval()
    with torch.no_grad():
        fake = generator(fixed_noise.to(device)).detach().cpu()
    # Denormalise [-1,1] → [0,1]
    fake = (fake + 1) / 2.0
    grid_path = os.path.join(SAMPLES_DIR, f"epoch_{epoch:04d}.png")
    save_image(fake, grid_path, nrow=nrow, padding=2, normalize=False)
    generator.train()
    return grid_path


# ══════════════════════════════════════════════════════════════════
# PLOT LOSS CURVES
# ══════════════════════════════════════════════════════════════════
def plot_losses(g_losses, d_losses, save_dir):
    if not MPL:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(g_losses, label="Generator Loss",     color="#A78BFA", linewidth=1.5)
    ax.plot(d_losses, label="Discriminator Loss", color="#F97316", linewidth=1.5)
    ax.set_title("GAN Training Loss Curves", fontsize=13, fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "gan_loss_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [✓] Loss curve saved → {path}")


# ══════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════
def train_gan():
    device = get_device()

    # ── Data ──────────────────────────────────────────────────────
    print(f"\n  Loading data from: {DATASET_PATH}")
    try:
        loader = get_dataloader(DATASET_PATH, IMG_SIZE, BATCH_SIZE)
    except Exception as e:
        print(f"  [ERROR] Could not load data: {e}")
        print(f"  Make sure the path exists: {DATASET_PATH}")
        sys.exit(1)

    # ── Models ────────────────────────────────────────────────────
    G = Generator(NOISE_DIM, G_FEATURES, CHANNELS).to(device)
    D = Discriminator(D_FEATURES, CHANNELS).to(device)

    G.apply(weights_init)
    D.apply(weights_init)

    print(f"\n  Generator parameters  : {sum(p.numel() for p in G.parameters()):,}")
    print(f"  Discriminator parameters: {sum(p.numel() for p in D.parameters()):,}")

    # ── Optimisers ────────────────────────────────────────────────
    opt_G = optim.Adam(G.parameters(), lr=LR_G, betas=(BETA1, BETA2))
    opt_D = optim.Adam(D.parameters(), lr=LR_D, betas=(BETA1, BETA2))

    # Binary Cross-Entropy loss
    criterion = nn.BCELoss()

    # Fixed noise for consistent sample visualisation across epochs
    fixed_noise = torch.randn(64, NOISE_DIM, 1, 1)

    # Label smoothing: real=0.9 (not 1.0), fake=0.0
    # Improves training stability (Improved GAN training tips)
    REAL_LABEL = 0.9
    FAKE_LABEL = 0.0

    g_losses, d_losses = [], []
    history = {"epoch": [], "G_loss": [], "D_loss": [], "D_real": [], "D_fake": []}

    print(f"\n  Training DCGAN for {EPOCHS} epochs ...")
    print(f"  Batch size: {BATCH_SIZE}  |  Noise dim: {NOISE_DIM}  |  Image size: {IMG_SIZE}×{IMG_SIZE}")
    print(f"  LR_G={LR_G}  LR_D={LR_D}  Beta1={BETA1}")
    print(f"  Samples → {SAMPLES_DIR}")
    print(f"  Checkpoints → {MODELS_DIR}")
    print("─" * 65)

    total_iters = 0

    for epoch in range(1, EPOCHS + 1):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_d_real = 0.0
        epoch_d_fake = 0.0
        n_batches = 0

        t0 = time.time()

        for batch_idx, data in enumerate(loader):
            # Get real images
            if isinstance(data, (list, tuple)):
                real_imgs = data[0].to(device)
            else:
                real_imgs = data.to(device)

            b_size = real_imgs.size(0)

            # ── TRAIN DISCRIMINATOR ──────────────────────────────
            # Goal: maximise D(real) and minimise D(G(noise))
            D.zero_grad()

            # Real images → D should output 1 (REAL_LABEL)
            real_labels = torch.full((b_size,), REAL_LABEL, device=device)
            out_real     = D(real_imgs)
            loss_D_real  = criterion(out_real, real_labels)
            loss_D_real.backward()
            D_x = out_real.mean().item()   # avg discriminator score on real

            # Fake images → D should output 0
            noise       = torch.randn(b_size, NOISE_DIM, 1, 1, device=device)
            fake_imgs   = G(noise)
            fake_labels = torch.full((b_size,), FAKE_LABEL, device=device)
            out_fake    = D(fake_imgs.detach())   # detach to not backprop into G
            loss_D_fake = criterion(out_fake, fake_labels)
            loss_D_fake.backward()
            D_G_z1 = out_fake.mean().item()

            loss_D = loss_D_real + loss_D_fake
            opt_D.step()

            # ── TRAIN GENERATOR ──────────────────────────────────
            # Goal: fool D — make D(G(noise)) → 1 (REAL_LABEL)
            G.zero_grad()

            # Re-evaluate fake images with UPDATED discriminator
            out_fake2  = D(fake_imgs)
            # Generator wants D to believe fakes are real
            loss_G     = criterion(out_fake2, real_labels)
            loss_G.backward()
            D_G_z2 = out_fake2.mean().item()
            opt_G.step()

            # Track losses
            epoch_g_loss += loss_G.item()
            epoch_d_loss += loss_D.item()
            epoch_d_real += D_x
            epoch_d_fake += D_G_z2
            n_batches    += 1
            total_iters  += 1

            g_losses.append(loss_G.item())
            d_losses.append(loss_D.item())

        # ── End of epoch ──────────────────────────────────────────
        avg_g = epoch_g_loss / n_batches
        avg_d = epoch_d_loss / n_batches
        avg_dr= epoch_d_real / n_batches
        avg_df= epoch_d_fake / n_batches
        elapsed = time.time() - t0

        print(f"  Epoch [{epoch:03d}/{EPOCHS}]  "
              f"G_loss={avg_g:.4f}  D_loss={avg_d:.4f}  "
              f"D(real)={avg_dr:.3f}  D(G)={avg_df:.3f}  "
              f"({elapsed:.0f}s)")

        history["epoch"].append(epoch)
        history["G_loss"].append(round(avg_g, 4))
        history["D_loss"].append(round(avg_d, 4))
        history["D_real"].append(round(avg_dr, 4))
        history["D_fake"].append(round(avg_df, 4))

        # Save sample images
        if epoch % SAMPLE_INTERVAL == 0 or epoch == 1:
            path = save_sample_grid(G, fixed_noise, epoch, device)
            print(f"  [✓] Samples saved → {path}")

        # Save checkpoint
        if epoch % SAVE_INTERVAL == 0 or epoch == EPOCHS:
            ckpt_path = os.path.join(MODELS_DIR, f"gan_generator_epoch{epoch:03d}.pth")
            torch.save({
                "epoch":           epoch,
                "generator_state": G.state_dict(),
                "discriminator_state": D.state_dict(),
                "opt_G_state":     opt_G.state_dict(),
                "opt_D_state":     opt_D.state_dict(),
                "noise_dim":       NOISE_DIM,
                "img_size":        IMG_SIZE,
                "g_loss":          avg_g,
                "d_loss":          avg_d,
            }, ckpt_path)
            print(f"  [✓] Checkpoint → {ckpt_path}")

    # ── Save final best generator ──────────────────────────────────
    final_path = os.path.join(MODELS_DIR, "gan_generator_final.pth")
    torch.save(G.state_dict(), final_path)
    print(f"\n  [✓] Final generator saved → {final_path}")

    # Save training history
    with open(os.path.join(MODELS_DIR, "gan_training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Plot loss curves
    plot_losses(g_losses, d_losses, GAN_OUTPUT_DIR)

    print("\n  ── Training Summary ─────────────────────────────────────")
    print(f"  Total epochs    : {EPOCHS}")
    print(f"  Final G loss    : {history['G_loss'][-1]:.4f}")
    print(f"  Final D loss    : {history['D_loss'][-1]:.4f}")
    print(f"  Generated samples in: {SAMPLES_DIR}")
    print(f"  Run with --mode generate to create new images\n")

    return G


# ══════════════════════════════════════════════════════════════════
# GENERATE NEW IMAGES (Inference)
# ══════════════════════════════════════════════════════════════════
def generate_images(num_images: int = 100, show_grid: bool = False):
    """
    Load saved generator and produce new AI-like images.
    Saves individual JPEGs to gan_outputs/generated_images/
    """
    device = get_device()

    # Find best checkpoint
    ckpt_path = os.path.join(MODELS_DIR, "gan_generator_final.pth")
    full_ckpt  = os.path.join(MODELS_DIR, "gan_generator_epoch{:03d}.pth".format(EPOCHS))

    # Try loading full checkpoint first (has all info)
    loaded_full = False
    for cp in [full_ckpt, os.path.join(MODELS_DIR, "gan_generator_final.pth")]:
        if os.path.exists(cp):
            try:
                ckpt = torch.load(cp, map_location=device)
                if isinstance(ckpt, dict) and "generator_state" in ckpt:
                    noise_dim = ckpt.get("noise_dim", NOISE_DIM)
                    img_size  = ckpt.get("img_size",  IMG_SIZE)
                    G = Generator(noise_dim, G_FEATURES, CHANNELS).to(device)
                    G.load_state_dict(ckpt["generator_state"])
                    loaded_full = True
                    print(f"  [✓] Loaded checkpoint: {cp}  (epoch={ckpt.get('epoch','?')})")
                elif isinstance(ckpt, dict) and not "generator_state" in ckpt:
                    # state dict only
                    G = Generator(NOISE_DIM, G_FEATURES, CHANNELS).to(device)
                    G.load_state_dict(ckpt)
                    loaded_full = True
                    print(f"  [✓] Loaded generator weights: {cp}")
                if loaded_full:
                    break
            except Exception as e:
                print(f"  [!] Could not load {cp}: {e}")

    if not loaded_full:
        print(f"\n  [ERROR] No trained generator found in {MODELS_DIR}")
        print("  Run: python gan_generator.py --mode train")
        sys.exit(1)

    G.eval()
    print(f"\n  Generating {num_images} images → {GENERATED_DIR}")

    from PIL import Image as PILImage

    batch_size = min(64, num_images)
    generated  = 0
    all_imgs   = []

    while generated < num_images:
        current_batch = min(batch_size, num_images - generated)
        noise = torch.randn(current_batch, NOISE_DIM, 1, 1, device=device)

        with torch.no_grad():
            fake = G(noise)

        # Denormalise [-1,1] → [0,255] uint8
        fake_np = ((fake.cpu().numpy().transpose(0, 2, 3, 1) + 1) / 2.0 * 255).astype(np.uint8)

        for j in range(current_batch):
            img_path = os.path.join(GENERATED_DIR, f"gan_gen_{generated+j+1:06d}.jpg")
            PILImage.fromarray(fake_np[j]).save(img_path, quality=95)
            if show_grid:
                all_imgs.append(fake[j].cpu())

        generated += current_batch
        print(f"  Progress: {generated}/{num_images}")

    print(f"\n  [✓] {num_images} images saved to {GENERATED_DIR}")

    # Save a preview grid
    if show_grid and all_imgs and MPL:
        n_show = min(64, len(all_imgs))
        grid = make_grid(all_imgs[:n_show], nrow=8, padding=2, normalize=True)
        grid_np = grid.permute(1, 2, 0).numpy()

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(grid_np)
        ax.axis("off")
        ax.set_title(f"GAN Generated Images — {n_show} samples", fontsize=14, fontweight="bold")
        plt.tight_layout()
        grid_path = os.path.join(GAN_OUTPUT_DIR, "generated_grid.png")
        plt.savefig(grid_path, dpi=150)
        plt.close()
        print(f"  [✓] Grid preview saved → {grid_path}")

    # Integration note
    print("\n  ── Integration with DWDM Pipeline ──────────────────────")
    print(f"  Generated images saved to: {GENERATED_DIR}")
    print("  These can be added to your AI training set to augment data.")
    print("  Run predict.py on these images to verify the classifier works:")
    print(f'  python predict.py --folder "{GENERATED_DIR}" --model resnet50\n')


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="DWDM: DCGAN — Train an AI image generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gan_generator.py --mode train
  python gan_generator.py --mode generate --num_images 100
  python gan_generator.py --mode generate --num_images 64 --show_grid
        """
    )
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "generate"],
                        help="train: train the GAN | generate: generate new images")
    parser.add_argument("--num_images", type=int, default=100,
                        help="(generate mode) number of images to generate")
    parser.add_argument("--show_grid", action="store_true",
                        help="(generate mode) save a preview grid PNG")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs")

    args = parser.parse_args()

    if args.epochs:
        global EPOCHS
        EPOCHS = args.epochs

    print(f"\n  Mode          : {args.mode}")
    print(f"  Dataset path  : {DATASET_PATH}")
    print(f"  Models dir    : {MODELS_DIR}")
    print(f"  GAN output    : {GAN_OUTPUT_DIR}")

    if args.mode == "train":
        train_gan()
    elif args.mode == "generate":
        generate_images(args.num_images, args.show_grid)


if __name__ == "__main__":
    main()