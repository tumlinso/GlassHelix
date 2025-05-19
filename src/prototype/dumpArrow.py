#!/usr/bin/env python3
"""
dumpArrowToU16Bin.py

Reads a HuggingFace Arrow dataset stream of fixed-length token lists
and writes them out as a contiguous binary of uint16 values.

Usage:
  ./dumpArrowToU16Bin.py \
      --arrow /path/to/dataset.arrow \
      --output /path/to/output_u16.bin
"""

import argparse
import pyarrow.ipc as ipc
import numpy as np
import time
import sys

FIXED_LEN = 2048
PAD_VALUE = np.uint16(1)
PROGRESS_STEP = 100_000

def dump_until_error(arrow_path: str, bin_path: str):
    print(f"[{time.strftime('%H:%M:%S')}] Opening Arrow: {arrow_path}", file=sys.stderr)
    total_recs = 0
    start_time = time.perf_counter()
    scratch = np.empty(FIXED_LEN, dtype=np.uint16)

    with open(arrow_path, "rb") as fa, open(bin_path, "wb") as fb:
        reader = ipc.open_stream(fa)

        # Read batches until StopIteration or an error occurs
        while True:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                break
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] Error reading batch, stopping: {e}", file=sys.stderr)
                break

            # Process each row in the batch
            list_arr = batch.column("input_ids")
            vals     = list_arr.values.to_numpy(zero_copy_only=False)
            offs     = list_arr.offsets.to_numpy()
            nrows    = len(offs) - 1

            for i in range(nrows):
                s, e = offs[i], offs[i+1]
                length = e - s
                row = vals[s:e].astype(np.uint16, copy=False)

                if length >= FIXED_LEN:
                    scratch[:] = row[:FIXED_LEN]
                else:
                    scratch[:length] = row
                    scratch[length:] = PAD_VALUE

                scratch.tofile(fb)
                total_recs += 1

                if total_recs % PROGRESS_STEP == 0:
                    elapsed = time.perf_counter() - start_time
                    print(f"[{time.strftime('%H:%M:%S')}] "
                          f"{total_recs:,} records written "
                          f"({FIXED_LEN} tokens each), {elapsed:.1f}s",
                          file=sys.stderr)

    total_vals  = total_recs * FIXED_LEN
    total_bytes = total_vals * 2
    print(f"[{time.strftime('%H:%M:%S')}] Done: wrote {total_recs:,} records "
          f"({total_vals:,} values, {total_bytes:,} bytes) to {bin_path}",
          file=sys.stderr)


if __name__ == "__main__":
    dump_until_error("/home/tumlinson/GlassHelix/data/genecorpus_30M_2048.dataset/dataset.arrow",
                     "/data/dataset.bin")