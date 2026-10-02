# Native inference diagnostic repairs

The actual LAPACK SVD right-nullspace code extracted columns from column-major VT. It now extracts each required VT row. Rectangular, tall, multiple-null-direction and zero matrices verify rank, orthonormality and independently recomputed scaled/physical-coordinate residuals. The added regression failed before the fix and passes afterward.

Finite candidate updates now retain authoritative normalized log weights. Display weights may underflow to zero without permanently excluding a finite-prior hypothesis; later evidence can restore it. An explicitly zero initial prior remains excluded. Evidence order and equivalent batching agree. Failed assimilation leaves the existing posterior and evidence history intact. This repairs the existing probabilistic strategy without changing the role of whole scientific alternatives.

Run `python3 -B docs/learning/check_native_repairs.py` to compile and execute both affected native regressions using LAPACK and the explicitly pinned installed Cellerator headers. This CPU check performs no CUDA execution. Independent source review found no material blocker; allocation-failure rollback was inspected by ordering but not fault-injected.
