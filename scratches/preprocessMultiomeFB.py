import scanpy as sc

if __name__ == "__main__":
    adata = sc.read_10x_mtx(
        '/home/tumlinson/GlassHelix/data/multiomicFB/HBr_15_A',
        var_names='gene_symbols',
        cache=True
    )

    # Basic filtering (customize thresholds as needed)
    sc.pp.filter_cells(adata, min_counts=1000)        # remove empty droplets
    sc.pp.filter_genes(adata, min_cells=3)            # remove lowly expressed genes

    # Normalize, log-transform, and find HVGs
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)

    # PCA & neighborhood graph
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)

    # UMAP embedding & Leiden clustering
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.5)  # cluster labels in adata.obs['leiden']

