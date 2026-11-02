"""
Project 36: Siamese Neural Network for Face Similarity (Contrastive Learning)
=============================================================================
Description:
Implementation of a Siamese Convolutional Architecture for open-set face verification
on the AT&T / Olivetti Faces dataset. Evaluates zero-shot identity verification using
Hadsell et al. (2006) Contrastive Loss, L2-normalized metric embeddings, ROC-AUC analysis,
Equal Error Rate (EER) threshold calibration, and bimodal distance distribution diagnostics.
"""

import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset

# -----------------------------------------------------------------------------
# Configuration & Seed Initialization
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
EMBEDDING_DIM = 128
MARGIN = 1.0
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4

NUM_SUBJECTS = 40
TRAIN_SUBJECTS = 30  # Zero-shot split: 30 subjects for training, 10 for evaluation
IMAGES_PER_SUBJECT = 10


def set_seed(seed: int = 42) -> None:
    """Ensure strict deterministic reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# Dataset: Paired Face Generator for Metric Learning
# -----------------------------------------------------------------------------
class FacePairDataset(Dataset):
    """
    Constructs balanced genuine (same subject) and impostor (different subjects)
    pairs from face image arrays for contrastive metric learning.
    Label 0 = Genuine (Similar), Label 1 = Impostor (Dissimilar).
    """
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        subject_ids: List[int],
        num_pairs: int = 1200,
        seed: int = 42,
    ):
        super().__init__()
        self.pairs = []
        self.pair_labels = []

        rng = np.random.RandomState(seed)

        # Organize images by subject ID
        subject_to_images = {sid: [] for sid in subject_ids}
        for img, lbl in zip(images, labels):
            if lbl in subject_to_images:
                subject_to_images[lbl].append(img)

        # Generate 50% Genuine pairs (Label 0) and 50% Impostor pairs (Label 1)
        half_pairs = num_pairs // 2

        # 1. Genuine pairs
        for _ in range(half_pairs):
            sid = rng.choice(subject_ids)
            idx1, idx2 = rng.choice(len(subject_to_images[sid]), size=2, replace=False)
            img1 = subject_to_images[sid][idx1]
            img2 = subject_to_images[sid][idx2]
            self.pairs.append((img1, img2))
            self.pair_labels.append(0.0)

        # 2. Impostor pairs
        for _ in range(half_pairs):
            sid1, sid2 = rng.choice(subject_ids, size=2, replace=False)
            idx1 = rng.choice(len(subject_to_images[sid1]))
            idx2 = rng.choice(len(subject_to_images[sid2]))
            img1 = subject_to_images[sid1][idx1]
            img2 = subject_to_images[sid2][idx2]
            self.pairs.append((img1, img2))
            self.pair_labels.append(1.0)

        # Shuffle pairs
        perm = rng.permutation(len(self.pairs))
        self.pairs = [self.pairs[i] for i in perm]
        self.pair_labels = [self.pair_labels[i] for i in perm]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img1, img2 = self.pairs[idx]
        label = self.pair_labels[idx]

        # Convert 64x64 float32 to [1, 64, 64] torch tensor normalized to [-1, 1]
        t1 = torch.from_numpy(img1).unsqueeze(0).float() * 2.0 - 1.0
        t2 = torch.from_numpy(img2).unsqueeze(0).float() * 2.0 - 1.0
        lbl = torch.tensor(label, dtype=torch.float32)

        return t1, t2, lbl


# -----------------------------------------------------------------------------
# Siamese Convolutional Backbone with L2 Embedding Normalization
# -----------------------------------------------------------------------------
class SiameseFaceNet(nn.Module):
    """
    Deep Convolutional Siamese architecture mapping 64x64 face images into
    a unit hyperspherical metric embedding space (L2 normalized).
    """
    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.conv_backbone = nn.Sequential(
            # Stage 1: 64x64 -> 32x32
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),

            # Stage 2: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),

            # Stage 3: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),

            # Stage 4: 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.projection_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim),
        )

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts and L2-normalizes the metric embedding for a single image."""
        feat = self.conv_backbone(x)
        emb = self.projection_head(feat)
        return F.normalize(emb, p=2, dim=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Processes a pair through shared-weight twin backbones."""
        emb1 = self.forward_one(x1)
        emb2 = self.forward_one(x2)
        return emb1, emb2


# -----------------------------------------------------------------------------
# Contrastive Loss Function (Hadsell et al., 2006)
# -----------------------------------------------------------------------------
class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss function:
    L(e1, e2, y) = (1 - y) * 0.5 * D^2 + y * 0.5 * max(0, margin - D)^2
    where y = 0 for genuine pairs and y = 1 for impostor pairs.
    """
    def __init__(self, margin: float = MARGIN):
        super().__init__()
        self.margin = margin

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # Pairwise Euclidean Distance
        dist = F.pairwise_distance(emb1, emb2, keepdim=True)

        # Genuine pair loss: pulls embeddings together
        loss_genuine = (1.0 - label.view(-1, 1)) * 0.5 * torch.pow(dist, 2)

        # Impostor pair loss: repels embeddings beyond margin
        loss_impostor = label.view(-1, 1) * 0.5 * torch.pow(
            torch.clamp(self.margin - dist, min=0.0), 2
        )

        return torch.mean(loss_genuine + loss_impostor)


# -----------------------------------------------------------------------------
# Training and Evaluation Engine
# -----------------------------------------------------------------------------
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    total = 0

    for x1, x2, label in dataloader:
        x1, x2, label = x1.to(device), x2.to(device), label.to(device)

        optimizer.zero_grad()
        emb1, emb2 = model(x1, x2)
        loss = criterion(emb1, emb2, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x1.size(0)
        total += x1.size(0)

    return running_loss / total


def evaluate_verification(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray, float, float]:
    """
    Evaluates face verification on held-out test identities.
    Computes validation loss, pairwise distances, ROC-AUC, and Equal Error Rate (EER).
    """
    model.eval()
    running_loss = 0.0
    total = 0
    all_distances = []
    all_labels = []

    with torch.no_grad():
        for x1, x2, label in dataloader:
            x1, x2, label = x1.to(device), x2.to(device), label.to(device)
            emb1, emb2 = model(x1, x2)
            loss = criterion(emb1, emb2, label)

            dist = F.pairwise_distance(emb1, emb2).cpu().numpy()
            all_distances.extend(dist)
            all_labels.extend(label.cpu().numpy())

            running_loss += loss.item() * x1.size(0)
            total += x1.size(0)

    val_loss = running_loss / total
    distances = np.array(all_distances)
    labels = np.array(all_labels)

    # In our convention, label=0 is Genuine, label=1 is Impostor.
    # Scores for ROC: larger distance implies more likely Impostor (label 1).
    auc = roc_auc_score(labels, distances)
    fpr, tpr, thresholds = roc_curve(labels, distances)
    fnr = 1.0 - tpr

    # Equal Error Rate (point where FPR == FNR)
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    optimal_threshold = thresholds[eer_idx]

    return val_loss, distances, labels, auc, eer, optimal_threshold


# -----------------------------------------------------------------------------
# Diagnostic Plotting Dashboard
# -----------------------------------------------------------------------------
def plot_siamese_dashboard(
    history: Dict[str, List[float]],
    distances: np.ndarray,
    labels: np.ndarray,
    auc: float,
    eer: float,
    optimal_threshold: float,
    test_dataset: FacePairDataset,
    model: nn.Module,
    device: torch.device,
    save_path: Path,
) -> None:
    """Renders a 4-panel publication-grade evaluation figure."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)

    # 1. Contrastive Loss Trajectory
    ax_loss = axes[0, 0]
    epochs = range(1, len(history["train_loss"]) + 1)
    ax_loss.plot(epochs, history["train_loss"], "o-", color="#1f77b4", linewidth=2.0, label="Train Loss")
    ax_loss.plot(epochs, history["val_loss"], "s--", color="#d62728", linewidth=2.0, label="Val Loss (Held-out IDs)")
    ax_loss.set_title("(a) Contrastive Loss Trajectory (Margin m=1.0)", fontsize=13, fontweight="bold", pad=10)
    ax_loss.set_xlabel("Epoch", fontsize=11)
    ax_loss.set_ylabel("Loss", fontsize=11)
    ax_loss.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)
    ax_loss.grid(True, linestyle=":", alpha=0.6)

    # 2. Bimodal Distance Distribution (Genuine vs Impostor)
    ax_dist = axes[0, 1]
    genuine_dist = distances[labels == 0.0]
    impostor_dist = distances[labels == 1.0]

    bins = np.linspace(0.0, 2.0, 35)
    ax_dist.hist(genuine_dist, bins=bins, alpha=0.65, color="#2ca02c", edgecolor="black", label=f"Genuine Pairs (N={len(genuine_dist)})", density=True)
    ax_dist.hist(impostor_dist, bins=bins, alpha=0.65, color="#d62728", edgecolor="black", label=f"Impostor Pairs (N={len(impostor_dist)})", density=True)
    ax_dist.axvline(optimal_threshold, color="#333333", linestyle="--", linewidth=2.2, label=f"Optimal Threshold $\\tau^*={optimal_threshold:.2f}$")

    ax_dist.set_title("(b) Verification Distance Distribution ($L_2$ Space)", fontsize=13, fontweight="bold", pad=10)
    ax_dist.set_xlabel("Pairwise Euclidean Distance $\|e_1 - e_2\|_2$", fontsize=11)
    ax_dist.set_ylabel("Probability Density", fontsize=11)
    ax_dist.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)
    ax_dist.grid(True, linestyle=":", alpha=0.6)

    # 3. ROC Curve & Equal Error Rate (EER)
    ax_roc = axes[1, 0]
    fpr, tpr, _ = roc_curve(labels, distances)
    ax_roc.plot(fpr, tpr, color="#ff7f0e", linewidth=2.4, label=f"Siamese ROC (AUC = {auc:.4f})")
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess")
    ax_roc.plot([eer], [1.0 - eer], "ro", markersize=8, label=f"EER = {eer*100:.2f}%")

    ax_roc.set_title(f"(c) Receiver Operating Characteristic (AUC = {auc:.3f})", fontsize=13, fontweight="bold", pad=10)
    ax_roc.set_xlabel("False Acceptance Rate (FAR)", fontsize=11)
    ax_roc.set_ylabel("True Acceptance Rate (TAR = 1 - FRR)", fontsize=11)
    ax_roc.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)
    ax_roc.grid(True, linestyle=":", alpha=0.6)

    # 4. Qualitative Face Verification Gallery (6 sample test pairs)
    ax_gal = axes[1, 1]
    ax_gal.axis("off")
    ax_gal.set_title("(d) Open-Set Face Verification Predictions (Held-Out IDs)", fontsize=13, fontweight="bold", pad=10)

    sub_spec = axes[1, 1].get_subplotspec()
    inner_grid = fig.add_subfigure(sub_spec)
    sub_axes = inner_grid.subplots(3, 2)

    sample_indices = [2, 5, 12, 17, 24, 30]
    model.eval()

    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            t1, t2, lbl = test_dataset[idx]
            e1, e2 = model(t1.unsqueeze(0).to(device), t2.unsqueeze(0).to(device))
            d = F.pairwise_distance(e1, e2).item()

            is_genuine = (lbl.item() == 0.0)
            pred_genuine = (d < optimal_threshold)
            is_correct = (is_genuine == pred_genuine)

            row, col = i // 2, i % 2
            sub_ax = sub_axes[row, col]

            # Stitch two faces horizontally
            img1 = (t1.squeeze().numpy() + 1.0) / 2.0
            img2 = (t2.squeeze().numpy() + 1.0) / 2.0
            combined = np.hstack([img1, np.ones((64, 4)), img2])

            sub_ax.imshow(combined, cmap="gray")
            sub_ax.axis("off")

            tag = "GENUINE" if is_genuine else "IMPOSTOR"
            color = "#2ca02c" if is_correct else "#d62728"
            verdict = "MATCH" if pred_genuine else "DIFF"
            sub_ax.set_title(
                f"{tag} | d={d:.2f} [{verdict}]",
                fontsize=8.5,
                color=color,
                fontweight="bold",
                pad=2,
            )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[Artifact] Saved Siamese verification dashboard to: {save_path}")


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------
def main() -> None:
    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("Project 36: Siamese Neural Network for Face Similarity (Contrastive Loss)")
    print("=" * 75)
    print(f"Compute Device        : {device}")
    print(f"Dataset               : AT&T / Olivetti Faces (40 subjects, 10 images each)")
    print(f"Evaluation Mode       : Strict Open-Set Zero-Shot Verification (Disjoint Identities)")
    print(f"Identity Split        : {TRAIN_SUBJECTS} Train Subjects | {NUM_SUBJECTS - TRAIN_SUBJECTS} Held-Out Test Subjects")
    print(f"Metric Embedding Dim  : {EMBEDDING_DIM} (Hyperspherical L2-Normalized)")
    print(f"Contrastive Margin    : {MARGIN}")
    print(f"Epochs                : {NUM_EPOCHS} | Batch Size: {BATCH_SIZE}")
    print("-" * 75)

    # 1. Load Dataset
    faces = fetch_olivetti_faces(data_home=str(DATA_DIR), download_if_missing=False)
    images = faces.images
    labels = faces.target

    train_ids = list(range(TRAIN_SUBJECTS))
    test_ids = list(range(TRAIN_SUBJECTS, NUM_SUBJECTS))

    train_dataset = FacePairDataset(images, labels, subject_ids=train_ids, num_pairs=1200, seed=RANDOM_SEED)
    test_dataset = FacePairDataset(images, labels, subject_ids=test_ids, num_pairs=400, seed=RANDOM_SEED + 1)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Constructed Pairs     : {len(train_dataset)} Train Pairs | {len(test_dataset)} Test Pairs")
    print("-" * 75)

    # 2. Model & Optimizer
    model = SiameseFaceNet(embedding_dim=EMBEDDING_DIM).to(device)
    criterion = ContrastiveLoss(margin=MARGIN)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters      : {total_params:,}")
    print("-" * 75)

    # 3. Initial Baseline Evaluation
    init_loss, _, _, init_auc, init_eer, _ = evaluate_verification(model, test_loader, criterion, device)
    print(f"Pre-training Baseline -> Loss: {init_loss:.4f} | ROC-AUC: {init_auc:.4f} | EER: {init_eer*100:.2f}%")
    print("-" * 75)

    # 4. Training Loop
    history = {"train_loss": [], "val_loss": [], "auc": [], "eer": []}

    start_time = time.perf_counter()
    for epoch in range(1, NUM_EPOCHS + 1):
        ep_start = time.perf_counter()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, dists, lbls, auc, eer, opt_thresh = evaluate_verification(model, test_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["auc"].append(auc)
        history["eer"].append(eer)

        ep_duration = time.perf_counter() - ep_start
        print(
            f"Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"ROC-AUC: {auc:.4f} | EER: {eer*100:5.2f}% | "
            f"Thresh: {opt_thresh:.2f} | Time: {ep_duration:.2f}s"
        )

    total_time = time.perf_counter() - start_time
    print("-" * 75)
    print(f"Training Completed in {total_time:.2f} seconds.")
    print(f"Final Verification ROC-AUC : {history['auc'][-1]:.4f} (Baseline: {init_auc:.4f})")
    print(f"Final Equal Error Rate (EER): {history['eer'][-1]*100:.2f}% (Baseline: {init_eer*100:.2f}%)")

    # 5. Save Model Checkpoint
    checkpoint_path = CHECKPOINT_DIR / "siamese_face_model.pt"
    torch.save(
        {
            "epoch": NUM_EPOCHS,
            "model_state_dict": model.state_dict(),
            "optimal_threshold": opt_thresh,
            "final_auc": history["auc"][-1],
            "final_eer": history["eer"][-1],
        },
        checkpoint_path,
    )
    print(f"\n[Checkpoint] Saved model weights to: {checkpoint_path}")

    # 6. Generate Publication-Quality Visual Dashboard
    dashboard_path = PROJECT_DIR / "siamese_face_verification_result.png"
    plot_siamese_dashboard(
        history=history,
        distances=dists,
        labels=lbls,
        auc=history["auc"][-1],
        eer=history["eer"][-1],
        optimal_threshold=opt_thresh,
        test_dataset=test_dataset,
        model=model,
        device=device,
        save_path=dashboard_path,
    )

    print("=" * 75)
    print("Siamese Network Experiment Completed Successfully.")
    print("=" * 75)


if __name__ == "__main__":
    main()

