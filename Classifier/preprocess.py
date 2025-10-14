import scanpy as sc
import pandas as pd

# Load Tabula Sapiens dataset
adata = sc.read_h5ad("/home/tumlinson/GlassHelix/data/tabulaSapiens/tabula-sapiens.h5ad")

# Extract raw count matrix (if present) or normalized data
if 'counts' in adata.layers:
    X_counts = adata.layers['counts']
else:
    X_counts = adata.X
# convert sparse matrix to dense if needed
try:
    X_counts = X_counts.toarray()
except AttributeError:
    pass

# Extract covariates: batch, tissue, donor
covariate_keys = ['batch', 'tissue', 'donor']
covariates = pd.get_dummies(adata.obs[covariate_keys].astype(str))

# Save extracted arrays and metadata
import numpy as np
np.save("data/X_counts.npy", X_counts)
np.save("data/covariates.npy", covariates.values)
adata.obs.to_csv("data/obs_metadata.csv")
adata.var.to_csv("data/var_metadata.csv")

