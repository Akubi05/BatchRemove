import os
import scanpy as sc 
import numpy as np
import scDML 
print(scDML.__version__)
import sys
print(sys.version)
from scDML import scDMLModel
from scDML.utils import print_dataset_information
# env scDML

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
OUTFILE = "scdml.npy"

N_TOP_GENES = 2000
N_PCS = 50

# read data
dataset="bct"
adata_raw=sc.read(H5AD)
print(adata_raw)
print_dataset_information(adata_raw,batch_key="BATCH",celltype_key="celltype")

#View raw data,check the batch effect of this batch effect 
adata_copy=adata_raw.copy()
sc.pp.normalize_total(adata_copy,target_sum=1e4)
sc.pp.log1p(adata_copy)
sc.pp.highly_variable_genes(adata_copy,n_top_genes=1000,subset=True)
sc.pp.scale(adata_copy)
sc.tl.pca(adata_copy)
sc.pp.neighbors(adata_copy)
scdml=scDMLModel(save_dir="./result/"+dataset+"/",verbose=True)
adata=scdml.preprocess(adata_raw,cluster_method="louvain",resolution=3.0,n_high_var=1000)

# visluzation of preprocessed adata
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pl.umap(adata,color=["BATCH","init_cluster"])

from scDML.utils import plotDendrogram,plotHeatMap
# convert adata to training data for neural network
scdml.convertInput(adata,batch_key="BATCH")
_,_,cor,_ = scdml.calculate_similarity()

from scDML.utils import plotSankey

ncluster_list=[10,8,6,4,3]# for example
merge_df=scdml.merge_cluster(ncluster_list=ncluster_list,merge_rule="rule2")
merge_df["celltype"]=adata.obs["celltype"].values

scdml.build_net()

# delete trained model to retrain the model
if os.path.isfile(os.path.join(scdml.save_dir,"scDML_model.pkl")):
    os.remove(os.path.join(scdml.save_dir,"scDML_model.pkl"))
    
features=scdml.train(expect_num_cluster=None)
adata.obsm["X_emb"]=features
adata.obs["reassign_cluster"]=scdml.train_label.astype(int).astype(str)
adata.obs["reassign_cluster"]=adata.obs["reassign_cluster"].astype("category")

sc.pp.neighbors(adata,use_rep="X_emb",random_state=0)
sc.tl.umap(adata)

# === save embedding & labels for later metrics ===
os.makedirs(OUTDIR, exist_ok=True)

# 1) 保存嵌入
emb_path = os.path.join(OUTDIR, OUTFILE)  
np.save(emb_path, adata.obsm["X_emb"].astype(np.float32))
print(f"[OK] Embedding saved -> {emb_path} shape={adata.obsm['X_emb'].shape}")

# 2) （可选）保存标签，便于之后直接评估
np.save(os.path.join(OUTDIR, "labels_celltype.npy"),
        adata.obs["celltype"].astype(str).to_numpy())
np.save(os.path.join(OUTDIR, "labels_reassign.npy"),
        adata.obs["reassign_cluster"].astype(str).to_numpy())


sc.pl.umap(adata,color=["celltype","BATCH"])
sc.pl.umap(adata,color=["celltype","reassign_cluster"],legend_loc="on data",legend_fontsize="xx-large")

from sklearn.metrics.cluster import adjusted_rand_score,normalized_mutual_info_score
ari=adjusted_rand_score(adata.obs["reassign_cluster"],adata.obs["celltype"])
nmi=normalized_mutual_info_score(adata.obs["reassign_cluster"],adata.obs["celltype"])
print("ARI={}".format(ari))
print("NMI={}".format(nmi))