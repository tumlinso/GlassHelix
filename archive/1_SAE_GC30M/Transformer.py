
# load dataset on remote
from datasets import load_dataset, DownloadConfig

ds = load_dataset(
    "ctheodoris/Genecorpus-30M",
    split="train",                            # grab the (only) split
    cache_dir="/home/tumlinson/data/genecorpus/",
    download_config=DownloadConfig(resume_download=True),
    streaming=False                            # ← ensure full, map‐style loading
)

# print dataset info
print(ds.features)
# Expected: {'input_ids': Sequence(Value('int32')), 'lengths': Value('int32')}
# print dataset size
example = ds[0]      # get the very first cell
print(example)
# → {'input_ids': [...], 'lengths': 2048}
