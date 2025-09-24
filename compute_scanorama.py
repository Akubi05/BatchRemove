# -*- coding: utf-8 -*-
# env new_env
"""
Scanorama batch correction (no argparse, only saves embeddings/Corrected.npy)

Pipeline:
1) 读取三个 h5ad，添加 batch 标签（仅用于记录/打印）
2) 每个批次：filter_cells/filter_genes -> normalize_total -> log1p -> HVG 子集
3) Scanorama 对齐/矫正，返回对齐后的表达矩阵（各批次列一致）
4) 纵向拼接所有批次的矫正表达矩阵 -> PCA -> 保存 Corrected.npy
"""

import os
import numpy as np
import scanpy as sc
from anndata import AnnData
from scipy.sparse import issparse
from sklearn.decomposition import PCA
import scanorama  # pip install scanorama

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
OUTFILE = "scanorama.npy"

N_TOP_GENES = 2000
N_PCS = 50
# ==============================================================


def preprocess_one(ad: AnnData, n_top: int) -> AnnData:
    """单批次基础预处理：去重基因名 -> 过滤 -> 归一化 -> log1p -> 选 HVG（子集）"""
    ad = ad.copy()
    ad.var_names_make_unique()
    sc.pp.filter_cells(ad, min_genes=200)
    sc.pp.filter_genes(ad, min_cells=3)
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=n_top, subset=True)
    return ad


def _to_dense_2d(a) -> np.ndarray:
    """把输入转成 2D dense float32（兼容 scipy.sparse / list / 1D）"""
    try:
        if issparse(a):
            a = a.toarray()
    except Exception:
        pass
    a = np.asarray(a)
    if a.ndim != 2:
        a = np.atleast_2d(a)
    return a.astype(np.float32, copy=False)


def scanorama_correct(ads: list[AnnData]) -> list[np.ndarray]:
    """用 Scanorama 对齐/矫正，返回每个批次的矫正表达矩阵（dense float32）列表"""
    datasets = []
    genes_list = []
    for ad in ads:
        X = ad.X.toarray() if issparse(ad.X) else np.asarray(ad.X)
        datasets.append(X.astype(np.float32, copy=False))
        genes_list.append(ad.var_names.tolist())

    corrected_list, _ = scanorama.correct(
        datasets, genes_list, return_dimred=False
    )
    # 统一为 2D dense float32，避免后续 vstack 报错
    corrected_list = [_to_dense_2d(c) for c in corrected_list]
    return corrected_list


def run_pca(X: np.ndarray, n_comp: int, seed: int = 0) -> np.ndarray:
    """对矫正表达矩阵做 PCA，返回 (n_cells, n_comp) 低维嵌入"""
    pca = PCA(n_components=n_comp, svd_solver="auto", random_state=seed)
    Z = pca.fit_transform(X)
    return Z.astype(np.float32, copy=False)


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
    random_state = 0
    # Print basic info
    if BATCH_KEY in adata.obs.columns:
        uniq = adata.obs[BATCH_KEY].astype(str).unique().tolist()
        print(f"[Info] {BATCH_KEY} 值: {uniq}")
    else:
        print(f"[Warn] 未在 adata.obs 中找到批次列 '{BATCH_KEY}'，将直接计算 PCA")

    # 每批次预处理
    ap = preprocess_one(adata, N_TOP_GENES)

    # Scanorama 矫正
    corrected_list = scanorama_correct([ap])

    # 合并为 (总细胞数, 对齐后基因数)
    X_corrected = np.vstack(corrected_list).astype(np.float32, copy=False)
    print(f"[OK] Scanorama corrected expression -> {X_corrected.shape} (cells x genes)")

    # 低维嵌入（PCA）
    Z = run_pca(X_corrected, N_PCS, seed=random_state)
    print(f"[OK] Corrected embedding (PCA {N_PCS}) -> {Z.shape}")

    os.makedirs(OUTDIR, exist_ok=True)
    out_path = os.path.join(OUTDIR, OUTFILE)
    np.save(out_path, Z)
    print(f"[OK] Saved -> {out_path} shape={Z.shape}")


if __name__ == "__main__":
    main()
