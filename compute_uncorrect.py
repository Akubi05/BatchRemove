# -*- coding: utf-8 -*-
# env : dachuang310
import os
import numpy as np
import scanpy as sc
from anndata import AnnData

# ===================== Config (edit here) =====================
# Mode A: single file (set H5AD to a path and leave INPUTS empty)
# H5AD = "bct_raw.h5ad"  
H5AD = "mural_raw.h5ad"
INPUTS = []
LABELS = None
# Mode B: multiple files (leave H5AD=None and list INPUTS)
# H5AD = None
# INPUTS = [
#     "neurips2021_s1d3.h5ad",
#     "neurips2021_s2d1.h5ad",
#     "neurips2021_s3d7.h5ad",
# ]
# LABELS = ["s1d3", "s2d1", "s3d7"]  # optional; if None, filenames (stem) will be used


# BATCH_KEY = "BATCH"
# CELL_TYPE = "celltype"
BATCH_KEY = "batch"
CELL_TYPE = "cell_type1"

OUTDIR = "embeddings"
OUTFILE = "uncorrect.npy"

N_TOP_GENES = 2000
N_PCS = 50
# ==============================================================


def embed_uncorrect(adata: AnnData, n_top_genes=2000, n_pcs=50) -> np.ndarray:
    """Preprocess, PCA, neighbors & UMAP on a fresh copy; return X_pca."""
    ad = adata.copy()
    sc.pp.filter_cells(ad, min_genes=200)
    sc.pp.filter_genes(ad, min_cells=3)
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=n_top_genes, subset=True)
    sc.pp.scale(ad, max_value=10)
    sc.tl.pca(ad, n_comps=n_pcs, svd_solver="arpack")

    # 邻居图 & UMAP（未做任何批次校正）
    sc.pp.neighbors(ad, n_neighbors=15, n_pcs=n_pcs)
    sc.tl.umap(ad)

    # 保存 UMAP（按批次 / 按细胞类型）
    sc.pl.umap(ad, color=BATCH_KEY, show=False, save="_uncorrect_after_batch.png")
    if CELL_TYPE in ad.obs.columns:
        sc.pl.umap(ad, color=CELL_TYPE, show=False, save="_uncorrect_after_celltype.png")

    return ad.obsm["X_pca"].copy()


# 多个文件，按文件名加载批次
def load_adata() -> AnnData:
    # Multi-file mode
    if H5AD is None and len(INPUTS) > 0:
        if LABELS is not None and len(LABELS) != len(INPUTS):
            raise ValueError("LABELS 的数量需与 INPUTS 文件数一致，或将 LABELS 设为 None 以使用文件名")
        adatas = []
        for i, pth in enumerate(INPUTS):
            if not os.path.exists(pth):
                raise FileNotFoundError(f"找不到输入文件: {pth}")
            ad = sc.read_h5ad(pth)
            label = LABELS[i] if LABELS is not None else os.path.splitext(os.path.basename(pth))[0]
            ad.obs[BATCH_KEY] = str(label)
            adatas.append(ad)
        adata = sc.concat(adatas, merge="same")
        return adata

    # Single-file mode
    elif H5AD is not None and len(INPUTS) == 0:
        if not os.path.exists(H5AD):
            raise FileNotFoundError(f"找不到输入文件: {H5AD}")
        return sc.read_h5ad(H5AD)

    else:
        raise ValueError("请在配置中二选一：设置 H5AD（单文件）或设置 INPUTS（多文件）；不要同时设置或同时为空。")


def main():
    adata = load_adata()

    # Print basic info
    if BATCH_KEY in adata.obs.columns:
        uniq = adata.obs[BATCH_KEY].astype(str).unique().tolist()
        print(f"[Info] {BATCH_KEY} 值: {uniq}")
    else:
        print(f"[Warn] 未在 adata.obs 中找到批次列 '{BATCH_KEY}'，将直接计算 PCA")

    # --- 用于对照的“未校正前”可视化（在 adata_pre 上做完整预处理） ---
    adata_pre = adata.copy()
    sc.pp.filter_cells(adata_pre, min_genes=200)
    sc.pp.filter_genes(adata_pre, min_cells=3)
    sc.pp.normalize_total(adata_pre, target_sum=1e4)
    sc.pp.log1p(adata_pre)
    sc.pp.highly_variable_genes(adata_pre, n_top_genes=N_TOP_GENES, subset=True)
    sc.pp.scale(adata_pre, max_value=10)
    sc.tl.pca(adata_pre, n_comps=N_PCS, svd_solver="arpack")
    sc.pp.neighbors(adata_pre, n_neighbors=15, n_pcs=N_PCS)
    sc.tl.umap(adata_pre)

    # 保存未校正 UMAP（按批次/细胞类型）
    sc.pl.umap(adata_pre, color=BATCH_KEY, show=False, save="_uncorrect_pre_batch.png")
    if CELL_TYPE in adata_pre.obs.columns:
        sc.pl.umap(adata_pre, color=CELL_TYPE, show=False, save="_uncorrect_pre_celltype.png")

    # --- 主流程：在“原始 adata”上调用 embed_uncorrect（避免二次 log1p/scale） ---
    X = embed_uncorrect(adata, n_top_genes=N_TOP_GENES, n_pcs=N_PCS)

    os.makedirs(OUTDIR, exist_ok=True)
    out_path = os.path.join(OUTDIR, OUTFILE)
    np.save(out_path, X)
    print(f"[OK] Saved -> {out_path} shape={X.shape}")


if __name__ == "__main__":
    main()
