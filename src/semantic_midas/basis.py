"""Fixed, identified linear lag basis; no beta-times-theta nonlinearity."""
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike, NDArray


def lag_basis(n_lags: int = 13, rank: int = 3) -> NDArray[np.float64]:
    """B[:,0]=1/K; other columns are zero-sum orthogonal lag contrasts.

    Every column has norm 1/sqrt(K). Positive first contrast weights recent
    observations more heavily. Coefficients are EFFECTIVE lag coefficients;
    higher-order coefficients are not separately normalized MIDAS weights.
    """
    if not 1 <= rank <= n_lags:
        raise ValueError("Require 1 <= rank <= n_lags")
    z = np.linspace(1., -1., n_lags)
    raw = np.polynomial.legendre.legvander(z, rank - 1)
    q, _ = np.linalg.qr(raw)
    for j in range(rank):
        if np.dot(q[:, j], raw[:, j]) < 0:
            q[:, j] *= -1
    return q / np.sqrt(n_lags)


def compress_lags(hf: ArrayLike, basis: ArrayLike | None = None,
                  mask: ArrayLike | None = None, intercept: bool = True) -> NDArray[np.float64]:
    """(..., factor, lag) -> (..., 1 + factor*rank).

    Missing slots use zero in already centered predictor units, without
    renormalizing kernels. Real inputs must be standardized on past data.
    This is an explicit imputation rule, not optimal missing-data inference.
    """
    a = np.asarray(hf, dtype=float)
    if a.ndim < 2:
        raise ValueError("hf requires factor and lag dimensions")
    b = lag_basis(a.shape[-1]) if basis is None else np.asarray(basis, dtype=float)
    if b.ndim != 2 or a.shape[-1] != b.shape[0]:
        raise ValueError("Lag dimension does not match basis")
    available = np.isfinite(a)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != a.shape:
            raise ValueError("mask and hf must have identical shapes")
        available &= m
    feats = np.einsum("...jk,kr->...jr", np.where(available, a, 0.), b)
    feats = feats.reshape(*a.shape[:-2], -1)
    if intercept:
        feats = np.concatenate([np.ones((*feats.shape[:-1], 1)), feats], axis=-1)
    return feats


def structural_map(n_factors: int = 3, rank: int = 3, q: int = 3) -> NDArray[np.float64]:
    """Frozen primitive-to-effective-coefficient map (not estimated on test).

    0: factor-1 total transmission; 1: factor-2 recency contrast;
    2: factor-3 total transmission with an opposite factor-2 spillover.
    """
    if n_factors < 3 or rank < 2 or q not in (1, 3):
        raise ValueError("This named map requires >=3 factors, >=2 basis terms, q in {1,3}")
    g = np.zeros((1 + n_factors * rank, 3))
    g[1, 0] = 1.
    g[1 + rank + 1, 1] = 1.
    g[1 + 2 * rank, 2] = 1. / np.sqrt(1.16)
    g[1 + rank, 2] = -0.4 / np.sqrt(1.16)
    return g[:, :q]
