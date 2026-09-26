#!/usr/bin/env python3
"""Standalone numerical reproducers for the 2026-09-26 GlassHelix source review.

These probes reproduce algorithms read through project-control. They DO NOT
load/build the GlassHelix repository or constitute an upstream regression run.
Python 3.10+, NumPy, and a system LAPACK library are required.

Usage: python review_probes.py [--json results.json]
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
from pathlib import Path

import numpy as np


def lapack_vt(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match the inspected Fortran DGESVD call: JOBU=N, JOBVT=A."""
    path = ctypes.util.find_library("lapack")
    if not path:
        raise RuntimeError("System LAPACK library not found; install LAPACK to run the SVD probe.")
    library = ctypes.CDLL(path)
    try:
        dgesvd = library.dgesvd_
    except AttributeError as exc:
        raise RuntimeError("The LAPACK library does not export dgesvd_.") from exc
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():
        raise ValueError("Expected a nonempty, finite, two-dimensional matrix.")
    rows, cols = matrix.shape
    a = np.array(matrix, dtype=np.float64, order="F", copy=True)
    singular = np.zeros(min(rows, cols), dtype=np.float64)
    u = np.zeros(1, dtype=np.float64)
    vt = np.zeros((cols, cols), dtype=np.float64, order="F")
    m, n = ctypes.c_int(rows), ctypes.c_int(cols)
    lda, ldu, ldvt = ctypes.c_int(rows), ctypes.c_int(1), ctypes.c_int(cols)
    jobu, jobvt = ctypes.c_char(b"N"), ctypes.c_char(b"A")
    info, lwork = ctypes.c_int(0), ctypes.c_int(-1)
    work = np.zeros(1, dtype=np.float64)
    ptr = ctypes.POINTER(ctypes.c_double)

    def execute() -> None:
        dgesvd(ctypes.byref(jobu), ctypes.byref(jobvt), ctypes.byref(m), ctypes.byref(n),
                a.ctypes.data_as(ptr), ctypes.byref(lda), singular.ctypes.data_as(ptr),
                u.ctypes.data_as(ptr), ctypes.byref(ldu), vt.ctypes.data_as(ptr),
                ctypes.byref(ldvt), work.ctypes.data_as(ptr), ctypes.byref(lwork),
                ctypes.byref(info))
        if info.value != 0:
            raise RuntimeError(f"DGESVD returned INFO={info.value}.")

    execute()
    lwork = ctypes.c_int(int(work[0]))
    work = np.zeros(lwork.value, dtype=np.float64)
    execute()
    return singular, vt


def nullspace_probe() -> dict:
    a = np.array([[1., 2., 3.], [2., 4., 6.]])
    singular, vt = lapack_vt(a)
    rank = int(np.sum(singular > 1e-10))
    n = a.shape[1]
    flat = vt.ravel(order="F")
    # Exact indexing pattern in the reviewed header (columns of VT).
    observed = np.stack([flat[i*n:(i+1)*n] for i in range(rank, n)])
    # A right singular vector is a ROW of VT: vt[i + j*ldvt] in Fortran storage.
    corrected = np.stack([[flat[i+j*n] for j in range(n)] for i in range(rank, n)])
    wrong_residual = np.linalg.norm(a @ observed.T, axis=0)
    right_residual = np.linalg.norm(a @ corrected.T, axis=0)
    assert np.max(wrong_residual) > 1.0
    assert np.max(right_residual) < 1e-10
    return {"source": "include/GlassHelix/interrogation/diagnostics.hh",
            "matrix": a.tolist(), "singular_values": singular.tolist(), "rank": rank,
            "reviewed_indexing_null_vectors": observed.tolist(),
            "reviewed_indexing_residual_norms": wrong_residual.tolist(),
            "corrected_indexing_null_vectors": corrected.tolist(),
            "corrected_indexing_residual_norms": right_residual.tolist()}


def normalize_log(values: np.ndarray) -> np.ndarray:
    maximum = float(np.max(values))
    if not np.isfinite(maximum):
        raise ValueError("No finite candidate log weight remains.")
    return values - (maximum + np.log(np.sum(np.exp(values - maximum))))


def weights_probe() -> dict:
    means = np.array([0., 1.])
    sigma = .02

    def likelihood(y: float) -> np.ndarray:
        return -.5 * ((y-means)/sigma)**2 - np.log(sigma) - .5*np.log(2*np.pi)

    def reviewed(sequence: tuple[float, ...]) -> list[float]:
        weights = np.array([.5, .5])
        for y in sequence:
            with np.errstate(divide="ignore"):
                logs = np.log(weights) + likelihood(y)
            weights = np.exp(normalize_log(logs))
        return weights.tolist()

    def persistent_logs(sequence: tuple[float, ...]) -> list[float]:
        logs = np.log(np.array([.5, .5]))
        for y in sequence:
            logs = normalize_log(logs + likelihood(y))
        return np.exp(logs).tolist()

    a, b = reviewed((0., 1.)), reviewed((1., 0.))
    corrected = persistent_logs((0., 1.))
    assert a == [1., 0.] and b == [0., 1.]
    assert np.allclose(corrected, [.5, .5])
    return {"source": "include/GlassHelix/inference/inference.hh",
            "means": means.tolist(), "sigma": sigma, "initial_weights": [.5, .5],
            "reviewed_update_observe_0_then_1": a,
            "reviewed_update_observe_1_then_0": b,
            "persistent_log_weights_observe_0_then_1": corrected,
            "explanation": "Equal total evidence for the two fixed candidates becomes order-dependent after a finite log weight is stored as numerical zero."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = {"review_date": "2026-09-26", "scope": "standalone reproductions, not upstream tests",
              "glasshelix_head": "da97825a18206f1ec727ea59439ca694995d7b94",
              "nullspace": nullspace_probe(), "candidate_weights": weights_probe()}
    output = json.dumps(result, indent=2)
    print(output)
    if args.json:
        args.json.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
