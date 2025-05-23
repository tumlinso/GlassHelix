import scanpy as sc
import pandas as pd
import os

# directory containing the files
directory = "/home/tumlinson/GlassHelix/data/tabulaSapiens/"

# get files in the directory
files = os.listdir(directory)
# filter for .h5ad files
h5ad_files = [f for f in files if f.endswith('.h5ad')]

# print the list of files
print("List of .h5ad files:")
for f in h5ad_files:
    print(f)