import os
import glob
import scanpy as sc
import numpy as np
from scipy import sparse
import struct

# 1) Find your files
input_dir   = "/home/tumlinson/GlassHelix/data/tabulaSapiens/"
pattern     = os.path.join(input_dir, "*.h5ad")
file_list   = glob.glob(pattern)
if not file_list:
    raise FileNotFoundError(f"No .h5ad files in {input_dir}")

# 2) Load gene names from the first file (no big data pulled into RAM)
ad0        = sc.read_h5ad(file_list[0], backed="r")
gene_names = list(ad0.var_names)  # capture order

# 3) Prepare containers
matrices        = []
cell_ids        = []
cell_ontologies = []

# 4) Loop (here first 5 files for a quick smoke-test)
for fn in file_list[:5]:
    ad = sc.read_h5ad(fn, backed="r")

    # 4a) Assert same gene order
    if list(ad.var_names) != gene_names:
        raise ValueError(f"Gene names mismatch in {fn}")

    # 4b) Pull just the CSR for log_normalized
    if "log_normalized" not in ad.layers:
        raise KeyError(f"'log_normalized' layer missing in {fn}")
    X_csr = sparse.csr_matrix(ad.layers["log_normalized"])

    # 4c) Collect
    matrices.append(X_csr)
    cell_ids.extend(ad.obs_names.tolist())
    cell_ontologies.extend(ad.obs["cell_ontology"].astype(str).tolist())

# 5) Stack all blocks – columns already aligned
big = sparse.vstack(matrices, format="csr")
rows, cols = big.shape
nnz        = big.nnz

h_rowptr = big.indptr.astype(np.int32)
h_colind = big.indices.astype(np.int32)
h_vals   = big.data.astype(np.float32)

# 6) Dump to your binary CSR
with open("all_expr.csrbin", "wb") as f:
    f.write(struct.pack("3i", rows, cols, nnz))
    f.write(h_rowptr.tobytes())
    f.write(h_colind.tobytes())
    f.write(h_vals.tobytes())

# 7) Dump cell‐mapping
with open("cell_mapping.tsv", "w") as f:
    f.write("new_id\torig_cell_id\tontology\n")
    for new_id, (orig, ont) in enumerate(zip(cell_ids, cell_ontologies), start=1):
        f.write(f"{new_id}\t{orig}\t{ont}\n")