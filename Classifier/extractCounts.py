import scanpy as sc
import pandas as pd
import os

# test on single file
filename ="/Users/gavintumlinson/Downloads/Ear_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad"

# Load the AnnData object
adata = sc.read_h5ad(filename)

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

# print head of the matrix
print(X_counts[:5, :5])
