# -*- coding: utf-8 -*-
"""
Pipeline: Scanpy preprocessing + Disentangled Autoencoder + MNN-based batch correction (Optimized)
Input : Multiple scRNA-seq batches (AnnData list or paths)
Output: Batch-corrected expression matrix (reconstructed from batch-free content)

Upgrades over baseline:
1) Encoder/Decoder blocks use LayerNorm + GELU + Dropout for stability on high-dim sparse data
2) GRL (gradient reversal) strength warm-up schedule during training (DANN-style S-curve)
3) Explicit cross-covariance penalty between z_c and z_b (decorrelation)
4) z_c centroid alignment across batches (encourages batch-invariant content)
5) BatchHead stabilized via spectral normalization; z-vectors L2-normalized before heads
6) Optional center loss on z_b to tighten per-batch clusters
7) Weight init: Xavier on Linear layers; weight decay in Adam

You can tune loss weights in train_disentae(...).
"""

import os
import math
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import spectral_norm
from torch.nn.functional import normalize
from sklearn.neighbors import NearestNeighbors
import umap  # for possible external UMAP usages

# ------------------------------
# Utility: Gradient Reversal Layer (for adversarial content loss)
# ------------------------------
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

# DANN-style warm-up schedule for GRL lambda
def grl_lambda_schedule(step: int, max_steps: int, alpha: float = 6.0, warmup: float = 0.1) -> float:
    p = min(1.0, step / max_steps)
    q = max(0.0, (p - warmup) / (1.0 - warmup))  # 前10%几乎不对抗
    return 2.0 / (1.0 + np.exp(-alpha * q)) - 1.0

# ------------------------------
# Disentangled Autoencoder (3-layer encoder/decoder) + improvements
# ------------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=(1024, 256), zc_dim=32, zb_dim=16, p_dropout=0.1):
        super().__init__()
        h1, h2 = hidden_dims
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, h1), nn.LayerNorm(h1), nn.GELU(), nn.Dropout(p_dropout),
            nn.Linear(h1, h2),        nn.LayerNorm(h2), nn.GELU(), nn.Dropout(p_dropout),
        )
        self.fc_zc = nn.Linear(h2, zc_dim)  # content
        self.fc_zb = nn.Linear(h2, zb_dim)  # batch noise
        self.drop_zc = nn.Dropout(p_dropout)

    def forward(self, x):
        h = self.backbone(x)
        zc = self.fc_zc(h)
        zb = self.fc_zb(h)
        if self.training:
            zc = self.drop_zc(zc)
        return zc, zb

class Decoder(nn.Module):
    def __init__(self, zc_dim=32, zb_dim=16, hidden_dims=(256, 1024), output_dim=2000, p_dropout=0.1):
        super().__init__()
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(zc_dim + zb_dim, h1), nn.LayerNorm(h1), nn.GELU(), nn.Dropout(p_dropout),
            nn.Linear(h1, h2),              nn.LayerNorm(h2), nn.GELU(), nn.Dropout(p_dropout),
            nn.Linear(h2, output_dim),
        )

    def forward(self, zc, zb):
        z = torch.cat([zc, zb], dim=1)
        x_hat = self.net(z)
        return x_hat

class BatchHead(nn.Module):
    """Batch classifier used for adversarial loss on z_c, and positive supervision on z_b."""
    def __init__(self, in_dim, n_batch, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(in_dim, hidden)), nn.ReLU(),
            spectral_norm(nn.Linear(hidden, n_batch))
        )
    def forward(self, z):
        return self.net(z)

# ------------------------------
# Training utils
# ------------------------------
@torch.no_grad()
def _std(x):
    return x.std(dim=0, keepdim=True) + 1e-6

# xcov penalty: encourage zc ⟂ zb
def xcov_loss(zc: torch.Tensor, zb: torch.Tensor) -> torch.Tensor:
    zc = zc - zc.mean(0, keepdim=True)
    zb = zb - zb.mean(0, keepdim=True)
    N = max(1, zc.size(0) - 1)
    cov = (zc.T @ zb) / N
    return (cov ** 2).mean()

# zc centroid alignment across batches
def zc_centroid_align(zc: torch.Tensor, batch_ids: torch.Tensor) -> torch.Tensor:
    means = []
    for bid in batch_ids.unique():
        means.append(zc[batch_ids == bid].mean(0, keepdim=True))
    if len(means) <= 1:
        return torch.tensor(0.0, device=zc.device)
    M = torch.cat(means, 0)  # [n_batch, zc_dim]
    return M.var(0, unbiased=False).mean()

# Optional centers for z_b (tight per-batch clusters)
class BatchCenters(nn.Module):
    def __init__(self, n_batch: int, dim: int, momentum: float = 0.9):
        super().__init__()
        self.register_buffer('centers', torch.zeros(n_batch, dim))
        self.momentum = momentum

    @torch.no_grad()
    def update(self, zb: torch.Tensor, batch_ids: torch.Tensor):
        for b in batch_ids.unique():
            m = zb[batch_ids == b].mean(0)
            self.centers[b] = self.momentum * self.centers[b] + (1 - self.momentum) * m

def center_loss(zb: torch.Tensor, batch_ids: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    diffs = zb - centers[batch_ids]
    return (diffs ** 2).mean()

class DisentAE(nn.Module):
    def __init__(self, input_dim, n_batch, zc_dim=32, zb_dim=16, p_dropout=0.1):
        super().__init__()
        self.encoder = Encoder(input_dim, (1024, 256), zc_dim, zb_dim, p_dropout)
        self.decoder = Decoder(zc_dim, zb_dim, (256, 1024), input_dim, p_dropout)
        self.batch_head_c = BatchHead(zc_dim, n_batch)  # adversarial via GRL
        self.batch_head_b = BatchHead(zb_dim, n_batch)  # encourage batch info

    def forward(self, x, grl_lambda=1.0, inject_noise_std=0.1):
        zc, zb = self.encoder(x)
        # inject random noise into batch channel during training
        if self.training and inject_noise_std > 0:
            zb = zb + inject_noise_std * torch.randn_like(zb)
        # reconstruction
        x_hat = self.decoder(zc, zb)
        # L2-normalized embeddings for heads to prevent scale hacks
        zcn = normalize(zc, p=2, dim=1)
        zbn = normalize(zb, p=2, dim=1)
        # batch predictions
        logits_c = self.batch_head_c(grad_reverse(zcn, grl_lambda))  # adversarial
        logits_b = self.batch_head_b(zbn)                            # supervised
        return x_hat, logits_c, logits_b, zc, zb

# Xavier init for all Linear layers
def _xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def train_disentae(
    X,
    batch_labels,
    *,
    zc_dim=64,
    zb_dim=24,
    batch_size=256,
    epochs=50,
    lr=1e-3,
    device="cpu",
    inject_noise_std=0.05,
    # loss weights
    w_adv=0.70,          # weight for adversarial CE on z_c
    w_b=0.9,            # weight for supervised CE on z_b
    w_xcov=0.03,         # weight for cross-cov decorrelation
    w_align=0.15,        # weight for zc centroid alignment
    w_center=0.05,      # weight for z_b center loss
    grl_alpha=8.0,     # steepness of GRL schedule
    weight_decay=5e-5,
    p_dropout=0.20,
):
    # X shape [N, D] float32; batch_labels [N] long
    X = torch.tensor(X, dtype=torch.float32)
    b = torch.tensor(batch_labels, dtype=torch.long)
    dataset = TensorDataset(X, b)
   # 在创建 DataLoader 之前插入：
    counts = np.bincount(b.numpy())
    sample_w = 1.0 / counts[b.numpy()]
    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False)

    n_batch = int(b.max().item()) + 1
    model = DisentAE(X.shape[1], n_batch, zc_dim, zb_dim, p_dropout=p_dropout).to(device)
    model.apply(_xavier_init)

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    recon_loss = nn.MSELoss()
    # 原: ce = nn.CrossEntropyLoss()
    class_w = torch.tensor(counts.max() / np.maximum(counts,1), dtype=torch.float32, device=device)
    ce = nn.CrossEntropyLoss(weight=class_w)

    # warm-up bookkeeping
    N = X.shape[0]
    steps_per_epoch = math.ceil(N / batch_size)
    max_steps = epochs * steps_per_epoch
    global_step = 0

    # center manager for z_b
    centers = BatchCenters(n_batch, zb_dim).to(device)

    model.train()
    for ep in range(epochs):
        Lr = Lc = Lb = Lxc = Lalign = Lctr = 0.0
        for x_i, b_i in loader:
            x_i, b_i = x_i.to(device), b_i.to(device)
            opt.zero_grad()

            lam = grl_lambda_schedule(global_step, max_steps, alpha=grl_alpha)
            x_hat, logits_c, logits_b, zc, zb = model(x_i, grl_lambda=lam, inject_noise_std=inject_noise_std)

            loss_recon = recon_loss(x_hat, x_i)
            loss_c = ce(logits_c, b_i)          # via GRL -> encoder sees -grad
            loss_b = ce(logits_b, b_i)
            loss_xc = xcov_loss(zc, zb)
            loss_al = zc_centroid_align(zc, b_i)
            centers.update(zb.detach(), b_i)
            loss_ctr = center_loss(zb, b_i, centers.centers)

            loss = (
                loss_recon + w_adv * loss_c + w_b * loss_b +
                w_xcov * loss_xc + w_align * loss_al + w_center * loss_ctr
            )
            loss.backward()
            opt.step()

            Lr += loss_recon.item(); Lc += loss_c.item(); Lb += loss_b.item()
            Lxc += loss_xc.item(); Lalign += loss_al.item(); Lctr += loss_ctr.item()
            global_step += 1

        avg = lambda S: S / steps_per_epoch
        rmse_per_gene = math.sqrt(avg(Lr))     # 因为 MSE 的平均已是逐元素
        print(f"Epoch {ep+1}/{epochs} | Recon {avg(Lr):.4f} | Adv(z_c) {avg(Lc):.4f} | "
            f"Batch(z_b) {avg(Lb):.4f} | XCov {avg(Lxc):.4f} | Align {avg(Lalign):.4f} | Center {avg(Lctr):.4f}")

    model.eval()
    with torch.no_grad():
        zc_all, zb_all = [], []
        for x_i, _ in DataLoader(dataset, batch_size=1024):
            x_i = x_i.to(device)
            zc, zb = model.encoder(x_i)
            zc_all.append(zc.cpu())
            zb_all.append(zb.cpu())
        Zc = torch.cat(zc_all).numpy()
        Zb = torch.cat(zb_all).numpy()
    return model, Zc, Zb

# ------------------------------
# Scanpy preprocessing across batches
# ------------------------------

def preprocess_batches(adata_list, *, min_genes=200, min_cells=3, n_top_genes=4000):
    """Per-batch QC, HVG intersection, normalize+log; returns stacked matrix and batch labels.
    adata_list: list of AnnData, each a batch
    """
    # 1) Per-batch QC
    for ad in adata_list:
        sc.pp.filter_cells(ad, min_genes=min_genes)
        sc.pp.filter_genes(ad, min_cells=min_cells)

    # 2) HVG per batch, then intersection
    hvg_sets = []
    for ad in adata_list:
        sc.pp.normalize_total(ad, target_sum=1e4)
        sc.pp.log1p(ad)
        sc.pp.highly_variable_genes(ad, n_top_genes=n_top_genes, flavor="seurat")
        hvg_sets.append(set(ad.var_names[ad.var["highly_variable"].values]))

    common_hvg = set.intersection(*hvg_sets)
    if len(common_hvg) < 200:
        raise ValueError("Common HVG set is too small; lower n_top_genes or check batch compatibility.")

    # Subset to HVG intersection and re-normalize (to keep pipeline clean and consistent)
    X_list, batch_ids = [], []
    for i, ad in enumerate(adata_list):
        ad_sub = ad[:, list(common_hvg)].copy()
        # Already normalized/logged; optionally scale
        sc.pp.scale(ad_sub, zero_center=True, max_value=10)
        X_list.append(ad_sub.X.toarray() if hasattr(ad_sub.X, 'toarray') else ad_sub.X)
        batch_ids.append(np.full(ad_sub.n_obs, i, dtype=int))

    X = np.vstack(X_list)
    batches = np.concatenate(batch_ids)
    return X, batches, list(common_hvg)

# ------------------------------
# MNN search & correction in content space Zc
# ------------------------------

def find_mnn_pairs(Z_ref, Z_tgt, k=50):
    # 在运行最近邻之前加：
    from sklearn.preprocessing import normalize as L2norm
    Z_ref = L2norm(Z_ref)    # 行向量 L2 归一
    Z_tgt = L2norm(Z_tgt)
    # Z_ref shape (N_ref, d)，Z_tgt shape (N_tgt, d)
    # 原: NearestNeighbors(n_neighbors=k).fit(...)
    nn_ref = NearestNeighbors(n_neighbors=k, metric='cosine').fit(Z_ref)
    nn_tgt = NearestNeighbors(n_neighbors=k, metric='cosine').fit(Z_tgt)
    idxs_ref = nn_ref.kneighbors(Z_tgt, return_distance=False)  # for each tgt -> k ref
    idxs_tgt = nn_tgt.kneighbors(Z_ref, return_distance=False)  # for each ref -> k tgt
    # Mutual check
    mnn_pairs = []
    # build reverse map for faster membership check
    tgt_neighbors_of_ref = [set(row.tolist()) for row in idxs_tgt]
    for j_tgt, ref_neighbors in enumerate(idxs_ref):
        for i_ref in ref_neighbors:
            if j_tgt in tgt_neighbors_of_ref[i_ref]:
                mnn_pairs.append((i_ref, j_tgt))
    return np.array(mnn_pairs, dtype=int)


def apply_mnn_correction_iterative(Zc_list, k=20):
    """Iteratively merge batches in Zc space using MNN-based shifts.
    Returns merged Zc and the per-batch corrected embeddings in original order.
    """
    # Choose reference as the largest batch
    sizes = [Z.shape[0] for Z in Zc_list]
    ref_idx = int(np.argmax(sizes))
    ref = Zc_list[ref_idx]
    order = list(range(len(Zc_list)))
    order.remove(ref_idx)

    corrected = [None] * len(Zc_list)
    corrected[ref_idx] = ref.copy()

    for tgt_idx in order:
        tgt = Zc_list[tgt_idx]
        pairs = find_mnn_pairs(ref, tgt, k=k)
        if pairs.size == 0:
            # If no pairs, skip shift
            corrected[tgt_idx] = tgt.copy()
            ref = np.vstack([ref, corrected[tgt_idx]])
            continue
        # Compute correction: for each target cell, average ref - tgt over its MNNs
        # Build per-target list of (ref_idx, tgt_idx) pairs
        diffs_sum = np.zeros_like(tgt)
        counts = np.zeros(tgt.shape[0])
        for i_ref, j_tgt in pairs:
            diffs_sum[j_tgt] += (ref[i_ref] - tgt[j_tgt])
            counts[j_tgt] += 1
        counts[counts == 0] = 1.0
        shift = diffs_sum / counts[:, None]
        tgt_corr = tgt + shift
        corrected[tgt_idx] = tgt_corr
        # Merge into reference for next iteration
        ref = np.vstack([ref, tgt_corr])

    # Build merged matrix following batch order
    return corrected

# ------------------------------
# Decode corrected embeddings back to expression
# ------------------------------
@torch.no_grad()
def decode_corrected_expression(model: DisentAE, Zc_batches, device="cpu"):
    Xcorr_batches = []
    for Zc in Zc_batches:
        zc = torch.tensor(Zc, dtype=torch.float32, device=device)
        zb_zero = torch.zeros((zc.shape[0], model.encoder.fc_zb.out_features), device=device)
        Xhat = model.decoder(zc, zb_zero)
        Xcorr_batches.append(Xhat.cpu().numpy())
    return Xcorr_batches

# ------------------------------
# Visualization helpers (optional)
# ------------------------------

def _factorize_labels(arr):
    arr = np.asarray(arr)
    if arr.dtype.kind in ("i", "u"):
        uniq = np.unique(arr)
        remap = {u:i for i,u in enumerate(uniq)}
        codes = np.vectorize(remap.get)(arr)
        names = [str(u) for u in uniq]
        return codes, names
    uniq, inv = np.unique(arr.astype(str), return_inverse=True)
    return inv, uniq.tolist()


def _encode_in_chunks(encoder, X, device="cpu", chunk=4096):
    zs = []
    enc_dev = next(encoder.parameters()).device
    use_device = torch.device(device) if device else enc_dev
    encoder = encoder.to(use_device)
    encoder.eval()
    with torch.no_grad():
        for s in range(0, X.shape[0], chunk):
            x_i = torch.tensor(X[s:s+chunk], dtype=torch.float32, device=use_device)
            zc, _ = encoder(x_i)   # 取内容通道
            zs.append(zc.detach().cpu().numpy())
    return np.vstack(zs)


def _scatter_umap(ax, emb2d, labels, names, title):
    k = len(np.unique(labels))
    cmap = "tab10" if k <= 10 else ("tab20" if k <= 20 else None)
    sc_ = ax.scatter(emb2d[:,0], emb2d[:,1], c=labels, cmap=cmap, s=6, linewidths=0)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    # 简洁图例（最多显示20类）
    if len(names) <= 20 and cmap is not None:
        import matplotlib.patches as mpatches
        uniq = np.unique(labels)
        colors = sc_.cmap(sc_.norm(uniq))
        handles = [mpatches.Patch(color=colors[i], label=names[u])
                   for i, u in enumerate(uniq)]
        ax.legend(handles=handles, bbox_to_anchor=(1.04,1), loc="upper left",
                  borderaxespad=0, fontsize=8)
