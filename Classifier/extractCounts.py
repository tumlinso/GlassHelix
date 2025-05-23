import os
import glob
import scanpy as sc
import numpy as np
from scipy import sparse
import struct

# Configuration (no argparse)
input_dir = "/home/tumlinson/GlassHelix/data/tabulaSapiens/"
test_files = 0  # set to 0 to process all files

# 1) Find your files
pattern     = os.path.join(input_dir, "*.h5ad")
file_list   = glob.glob(pattern)
if not file_list:
    raise FileNotFoundError(f"No .h5ad files in {input_dir}")

# 3a) Build file-to-category mapping
unique_files = sorted(file_list)
file_to_id = {fn: idx for idx, fn in enumerate(unique_files)}

# 2) Load gene names from the first file (no big data pulled into RAM)
ad0        = sc.read_h5ad(file_list[0], backed="r")
gene_names = list(ad0.var_names)  # capture order

# 3) Prepare containers
matrices         = []
orig_indices     = []
cell_ontologies  = []
file_categories  = []

# 4) Loop (here first 5 files for a quick smoke-test)
if test_files == 0:
    files_to_process = file_list
else:
    files_to_process = file_list[:test_files]

for fn in files_to_process:
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
    # record the original (file-local) cell indices
    orig_indices.extend(range(ad.n_obs))
    # record the file category for each cell
    file_categories.extend([file_to_id[fn]] * ad.n_obs)
    cell_ontologies.extend(ad.obs["cell_ontology_class"].astype(str).tolist())

# 5) Stack all blocks – columns already aligned

big = sparse.vstack(matrices, format="csr")
rows, cols = big.shape
nnz        = big.nnz
# 5a) Ensure dimensions and indices fit in uint32 without loss
max_uint32 = np.iinfo(np.uint32).max
if rows > max_uint32 or cols > max_uint32 or nnz > max_uint32:
    raise ValueError(f"Matrix too large for uint32: rows={rows}, cols={cols}, nnz={nnz}")
if big.indptr.min() < 0 or big.indptr.max() > max_uint32:
    raise ValueError(f"Row pointer out of uint32 bounds: min={big.indptr.min()}, max={big.indptr.max()}")
if big.indices.min() < 0 or big.indices.max() > max_uint32:
    raise ValueError(f"Column indices out of uint32 bounds: min={big.indices.min()}, max={big.indices.max()}")

h_rowptr = big.indptr.astype(np.uint32)
h_colind = big.indices.astype(np.uint32)
h_vals   = big.data.astype(np.float32)

# 6) Dump to your binary CSR
with open("/home/tumlinson/GlassHelix/data/tabulaSapiens/all_expr.csrbin", "wb") as f:
    f.write(struct.pack("3I", rows, cols, nnz))
    f.write(h_rowptr.tobytes())
    f.write(h_colind.tobytes())
    f.write(h_vals.tobytes())

# 7) Build unique ontology category mapping
unique_ont = sorted(set(cell_ontologies))
cat_to_id = {ont: idx for idx, ont in enumerate(unique_ont)}
ontology_ids = [cat_to_id[ont] for ont in cell_ontologies]

# 8) Dump the ontology category dictionary
with open("/home/tumlinson/GlassHelix/data/tabulaSapiens/ontology_categories.tsv", "w") as f_cat:
    f_cat.write("ontology_id\tontology_class\n")
    for idx, ont in enumerate(unique_ont):
        f_cat.write(f"{idx}\t{ont}\n")
print(f"Wrote ontology_categories.tsv with {len(unique_ont)} categories")

# 9) Dump the file categories dictionary
with open("/home/tumlinson/GlassHelix/data/tabulaSapiens/file_categories.tsv", "w") as f_filecat:
    f_filecat.write("file_id\tfilename\n")
    for fid, fname in enumerate(unique_files):
        f_filecat.write(f"{fid}\t{os.path.basename(fname)}\n")
print(f"Wrote file_categories.tsv with {len(unique_files)} entries")

# 10) Rewrite the cell-mapping to use ontology IDs and file IDs
with open("/home/tumlinson/GlassHelix/data/tabulaSapiens/cell_mapping.tsv", "w") as f:
    f.write("new_id\torig_cell_index\tontology_id\tfile_id\n")
    for new_id, (orig_idx, oid, fid) in enumerate(zip(orig_indices, ontology_ids, file_categories), start=1):
        f.write(f"{new_id}\t{orig_idx}\t{oid}\t{fid}\n")
print(f"Wrote cell_mapping.tsv with {len(orig_indices)} entries")