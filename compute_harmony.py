# -*- coding: utf-8 -*-
import os, numpy as np, scanpy as sc
import harmonypy
from anndata import AnnData

# ===================== Config (edit here) =====================
# Mode A: single file (set H5AD to a path and leave INPUTS empty)
# H5AD = "bct_raw.h5ad"
H5AD = "macaque_raw.h5ad"
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


BATCH_KEY = "BATCH"
CELL_TYPE = "celltype"
OUTDIR = "embeddings"
OUTFILE = "harmony.npy"

N_TOP_GENES = 2000
N_PCS = 50
# ==============================================================

def embed_harmony(adata: AnnData, batch_key: str, n_top_genes=2000, n_pcs=50) -> np.ndarray:
    ad = adata.copy()
    sc.pp.filter_cells(ad, min_genes=200)
    sc.pp.filter_genes(ad, min_cells=3)
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=n_top_genes, subset=True)
    sc.pp.scale(ad, max_value=10)
    sc.tl.pca(ad, n_comps=n_pcs, svd_solver="arpack")
    Z = ad.obsm["X_pca"]
    meta = ad.obs[[batch_key]].copy()
    ho = harmonypy.run_harmony(Z, meta, batch_key)
    X_corr = np.asarray(ho.Z_corr)          # 可能是 (n_pcs, n_cells)
    if X_corr.ndim == 1:                    # 极端情况被 squeeze 成一维
        X_corr = X_corr.reshape(-1, 1)
    if X_corr.shape[0] != ad.n_obs and X_corr.shape[1] == ad.n_obs:
        X_corr = X_corr.T                   # 转成 (n_cells, n_pcs)
    assert X_corr.shape[0] == ad.n_obs, f"after Harmony, got {X_corr.shape}, expected ({ad.n_obs}, k)"
    return ad, X_corr


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

    ad_hv, X = embed_harmony(adata, BATCH_KEY, n_top_genes=N_TOP_GENES, n_pcs=N_PCS)
    # 把矫正后的表示放进 .obsm
    ad_hv.obsm["X_pca_harmony"] = X
    # 基于矫正后的表示建图并跑 UMAP
    sc.pp.neighbors(ad_hv, use_rep="X_pca_harmony", n_neighbors=15)
    sc.tl.umap(ad_hv)
    # 保存“矫正后”的 UMAP（按批次 / 按细胞类型）
    sc.pl.umap(ad_hv, color=BATCH_KEY, show=False, save="_harmony_post_batch.png")
    if CELL_TYPE in ad_hv.obs.columns:
        sc.pl.umap(ad_hv, color=CELL_TYPE, show=False, save="_harmony_post_celltype.png")
    
    os.makedirs(OUTDIR, exist_ok=True)
    out_path = os.path.join(OUTDIR, OUTFILE)
    np.save(out_path, X)
    print(f"[OK] Saved -> {out_path} shape={X.shape}")


if __name__ == "__main__":
    main()
