==> Uncorrect: X shape = (30302, 50)
==> MyModel: X shape = (30302, 32)
==> Harmony: X shape = (30302, 50)
==> Scanorama: X shape = (30302, 50)
==> Seurat: X shape = (30302, 50)
ARI	NMI	ASW_celltype	ASW_batch	iLISI	KL
method						
Uncorrect	0.864393	0.870350	0.180905	0.495052	1.462206	1.059711
MyModel	0.110345	0.111975	-0.042026	0.503504	3.169371	0.175317
Harmony	0.939202	0.937747	0.213231	0.518558	3.291692	0.305639
Scanorama	0.424838	0.572141	0.112822	0.474185	1.341958	1.114245
Seurat	0.117358	0.128822	-0.026501	0.496428	1.972205	0.731611

                 scDML(louvain)
ARI                       0.934
NMI                       0.922
ASW_label                 0.783
ASW_label/batch           0.933
BatchKL                   0.410
cLISI                     1.000
iLISI                     2.364
保存: benchmark_out/metrics_summary.csv