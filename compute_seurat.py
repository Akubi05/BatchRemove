# -*- coding: utf-8 -*-
# env new_env
import os
import argparse
import numpy as np
import scanpy as sc
from anndata import AnnData
import pandas as pd
from scipy.sparse import issparse, coo_matrix

# ---- 限制线程，避免 MKL / OpenBLAS 冲突 ----
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import rpy2.robjects.numpy2ri as numpy2ri

# 启用自动转换（pandas / numpy -> R）
pandas2ri.activate()
numpy2ri.activate()


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
OUTFILE = "seurat.npy"

N_TOP_GENES = 2000
N_PCS = 50
# ==============================================================

def run_seurat_integration(adata: AnnData, batch_key: str, n_pcs=50) -> np.ndarray:
    """将 AnnData 原始 counts 和批次信息传到 R，使用 Seurat 做整合（RPCA），返回整合后的 PCA embedding。"""
    adata = adata.copy()
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    # 准备 counts 稀疏三元组（cells x genes -> COO）
    X = adata.X
    if not issparse(X):
        X = coo_matrix(X)
    else:
        X = X.tocoo()

    # Seurat 期望 genes x cells，因此：
    #   i: gene 索引（1-based），来自 X.col
    #   j: cell 索引（1-based），来自 X.row
    i = (X.col.astype(np.int32) + 1)
    j = (X.row.astype(np.int32) + 1)
    x = np.asarray(X.data, dtype=np.float64).ravel()  # 保证是一维向量

    n_genes = int(adata.n_vars)
    n_cells = int(adata.n_obs)
    gene_names = np.array(adata.var_names.astype(str))
    cell_names = np.array(adata.obs_names.astype(str))

    # 元数据：至少包含 batch
    meta_df = pd.DataFrame(
        {batch_key: adata.obs[batch_key].astype(str).values},
        index=adata.obs_names
    )

    # ---- R 端设置与变量注入 ----
    ro.r('Sys.setenv(RETICULATE_PYTHON="")')  # 避免 reticulate 嵌入 Python 导致 segfault
    ro.r('suppressMessages(library(Matrix))')
    ro.r('suppressPackageStartupMessages(library(Seurat))')
    ro.r('suppressMessages(library(future)); plan("sequential"); options(Seurat.num.cores = 1)')
    ro.r('if (requireNamespace("RhpcBLASctl", quietly=TRUE)) RhpcBLASctl::blas_set_num_threads(1)')

    ro.r.assign("i_idx", i)
    ro.r.assign("j_idx", j)
    ro.r.assign("x_vals", x)
    ro.r.assign("n_genes", n_genes)
    ro.r.assign("n_cells", n_cells)
    ro.r.assign("gene_names", gene_names)
    ro.r.assign("cell_names", cell_names)
    ro.r.assign("meta_df", pandas2ri.py2rpy(meta_df))
    ro.r.assign("batch_key", batch_key)
    ro.r.assign("npcs", int(n_pcs))

    # ---- 在 R 中执行 Seurat Integration（RPCA）----
    ro.r("""
    # 构建稀疏 counts（genes x cells）
    i_idx <- as.integer(i_idx)
    j_idx <- as.integer(j_idx)
    x_vals <- as.numeric(x_vals); dim(x_vals) <- NULL  # 去掉 array 的 dim
    counts <- sparseMatrix(i = i_idx, j = j_idx, x = x_vals, dims = c(n_genes, n_cells))
    dimnames(counts) <- list(gene_names, cell_names)

    so <- CreateSeuratObject(counts = counts)

    # 添加元数据（保证行名与细胞名一致）
    meta_df <- as.data.frame(meta_df)
    if (is.null(rownames(meta_df)) || any(rownames(meta_df) != colnames(so))) {
      rownames(meta_df) <- colnames(so)
      meta_df <- meta_df[colnames(so), , drop = FALSE]
    }
    so <- AddMetaData(so, metadata = meta_df)

    # 按批次拆分并预处理
    so.list <- SplitObject(so, split.by = batch_key)
    so.list <- lapply(so.list, function(x) {
      x <- NormalizeData(x, verbose = FALSE)
      x <- FindVariableFeatures(x, nfeatures = 2000, verbose = FALSE)
      x
    })

    # 选择整合特征
    features <- SelectIntegrationFeatures(object.list = so.list, nfeatures = 2000)

    # 为 RPCA 准备：每个子对象按 features 做 Scale + PCA
    so.list <- lapply(so.list, function(x) {
      x <- ScaleData(x, features = features, verbose = FALSE)
      x <- RunPCA(x, features = features, npcs = npcs, verbose = FALSE)
      x
    })

    # FindIntegrationAnchors with RPCA（更稳，避免 CCA 崩溃）
    anchors <- FindIntegrationAnchors(object.list = so.list,
                                      anchor.features = features,
                                      reduction = "rpca",
                                      dims = 1:npcs)

    integ <- IntegrateData(anchorset = anchors, dims = 1:npcs)

    DefaultAssay(integ) <- "integrated"
    integ <- ScaleData(integ, verbose = FALSE)
    integ <- RunPCA(integ, npcs = npcs, verbose = FALSE)

    pca_mat <- Embeddings(integ, reduction = "pca")
    """)

    # 取回整合后的 PCA
    pca_np = np.array(ro.r('pca_mat'))
    return pca_np

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

    


    # Seurat 批次整合（RPCA）
    X = run_seurat_integration(adata, batch_key=BATCH_KEY, n_pcs=N_PCS)
    print(f"[OK] Corrected PCA -> {X.shape}")

    # 保存
    os.makedirs(OUTDIR, exist_ok=True)
    out_path = os.path.join(OUTDIR, OUTFILE)
    np.save(out_path, X)
    print(f"[OK] Saved -> {out_path} shape={X.shape}")



if __name__ == "__main__":
    main()
