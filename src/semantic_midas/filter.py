"""Shared-covariance Kalman filter and exact mean-injection sensitivities.

Only H, released y, public break time, and fixed numerical parameters enter.
Neither true coefficients nor true break signs are inputs to this estimator.
Target-release delay uses an augmented state, never stale-y/current-state
substitution. Multiple arms can be reconstructed from one exact sensitivity
path; tests compare this to independently rerunning the actual filter.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class FilterConfig:
    q: float = 1e-4
    r: float = 0.25
    p0: float = 5.
    jump_var: float = 0.5625
    target_delay: int = 0

    def validate(self) -> None:
        if self.q < 0 or self.r <= 0 or self.p0 <= 0 or self.jump_var < 0:
            raise ValueError("Invalid process/measurement/initial/jump variance")
        if not isinstance(self.target_delay, int) or self.target_delay < 0:
            raise ValueError("target_delay must be a nonnegative integer")


@dataclass
class FilterResult:
    prediction: NDArray[np.float64]  # N x T, prediction before release at t
    response: NDArray[np.float64]    # N x T x d, derivative wrt one jump mean
    forecast_variance: NDArray[np.float64]
    gain_norm: NDArray[np.float64]
    final_mean: NDArray[np.float64]
    final_covariance: NDArray[np.float64]
    min_covariance_eigenvalue: float
    observed_counts: NDArray[np.int64]


def same_filter_paths(H: ArrayLike, y: ArrayLike, break_time: int,
                      G: ArrayLike, config: FilterConfig = FilterConfig(),
                      *, injection: ArrayLike | None = None,
                      injection_time: int | None = None,
                      jump_basis: ArrayLike | None = None,
                      check_psd: bool = False) -> FilterResult:
    """Batched exact random-walk coefficient KF with release-state augmentation.

    At origin t: propagate, apply common covariance reset / mean intervention,
    forecast y_t, THEN assimilate y_(t-delay), if released and nonmissing.
    Thus delay=0 is standard forecast-then-update, and delay=2 withholds the
    first post-break label until after the third post-break forecast.
    injection is the ONLY mean intervention; covariance does not depend on it.
    jump_basis optionally decouples the common covariance reset from the semantic
    map G. This is useful when G itself is learned from historical outcomes.
    """
    config.validate()
    h = np.asarray(H, dtype=float)
    obs = np.asarray(y, dtype=float)
    g = np.asarray(G, dtype=float)
    if h.ndim == 2:
        h = h[None, ...]
    if obs.ndim == 1:
        obs = obs[None, ...]
    if h.ndim != 3 or obs.shape != h.shape[:2] or not np.isfinite(h).all():
        raise ValueError("Require finite H[N,T,d] and matching y[N,T]")
    if np.isinf(obs).any():
        raise ValueError("y may contain NaN for missing releases, but not infinity")
    n, T, d = h.shape
    if g.ndim != 2 or g.shape[0] != d or not np.isfinite(g).all():
        raise ValueError("G must be a finite d by q matrix")
    if not 0 <= break_time < T:
        raise ValueError("break_time outside observations")
    inj_t = break_time if injection_time is None else injection_time
    if not 0 <= inj_t < T:
        raise ValueError("injection_time outside observations")
    inj = np.zeros((n, d)) if injection is None else np.asarray(injection, dtype=float)
    if inj.shape == (d,):
        inj = np.broadcast_to(inj, (n, d))
    if inj.shape != (n, d) or not np.isfinite(inj).all():
        raise ValueError("injection must have shape d or N,d and be finite")
    L = config.target_delay + 1
    D = L * d
    # Negative-time lag states represent the same initial unknown coefficient.
    base_p = np.kron(np.ones((L, L)), np.eye(d) * config.p0)
    P = np.broadcast_to(base_p, (n, D, D)).copy()
    m = np.zeros((n, D))
    sensitivity = np.zeros((n, D, d))
    pred = np.zeros((n, T)); response = np.zeros((n, T, d))
    variance = np.zeros((n, T)); gain_norm = np.zeros((n, T))
    counts = np.zeros((n, T), dtype=np.int64)
    seen = np.zeros(n, dtype=np.int64)
    min_eig = float("nan")
    shift_idx = np.r_[np.arange(d), np.arange(D - d)] if L > 1 else np.arange(d)
    jb = g if jump_basis is None else np.asarray(jump_basis, dtype=float)
    if jb.ndim != 2 or jb.shape[0] != d or not np.isfinite(jb).all():
        raise ValueError("jump_basis must be a finite d by r matrix")
    reset = config.jump_var * (jb @ jb.T)
    diag = np.arange(d)
    for t in range(T):
        if t > 0 and L > 1:
            m = m[:, shift_idx]
            P = P[:, shift_idx][:, :, shift_idx]
            sensitivity = sensitivity[:, shift_idx]
        P[:, diag, diag] += config.q
        if t == break_time:
            P[:, :d, :d] += reset
        if t == inj_t:
            m[:, :d] += inj
            sensitivity[:, :d, :] += np.eye(d)
        ht = h[:, t]
        pred[:, t] = np.einsum("ni,ni->n", ht, m[:, :d])
        response[:, t] = np.einsum("ni,nij->nj", ht, sensitivity[:, :d])
        variance[:, t] = np.einsum("ni,nij,nj->n", ht, P[:, :d, :d], ht) + config.r
        counts[:, t] = seen
        u = t - config.target_delay
        if u < 0:
            continue
        block = slice(config.target_delay * d, (config.target_delay + 1) * d)
        hu = h[:, u]
        ph = np.einsum("nij,nj->ni", P[:, :, block], hu)
        denom = np.einsum("ni,ni->n", hu, ph[:, block]) + config.r
        valid = np.isfinite(obs[:, u])
        K = (ph / denom[:, None]) * valid[:, None]
        innovation = np.where(valid, obs[:, u], 0.) - np.einsum("ni,ni->n", hu, m[:, block])
        innovation = np.where(valid, innovation, 0.)
        m += K * innovation[:, None]
        hs = np.einsum("ni,nij->nj", hu, sensitivity[:, block])
        sensitivity -= K[:, :, None] * hs[:, None, :]
        # Exact scalar Joseph simplification, followed by symmetry restoration.
        P -= K[:, :, None] * ph[:, None, :]
        P = 0.5 * (P + np.swapaxes(P, 1, 2))
        gain_norm[:, t] = np.linalg.norm(K, axis=1)
        seen += valid
        if check_psd:
            eig = float(np.linalg.eigvalsh(P).min())
            min_eig = eig if np.isnan(min_eig) else min(min_eig, eig)
            if eig < -1e-8:
                raise FloatingPointError(f"Covariance lost PSD at t={t}: {eig}")
    if not (np.isfinite(pred).all() and np.isfinite(P).all()):
        raise FloatingPointError("Nonfinite filter output")
    if not check_psd:
        min_eig = float(np.linalg.eigvalsh(P).min())
    return FilterResult(pred, response, variance, gain_norm, m, P, min_eig, counts)


def loss_identity(y: ArrayLike, zero: ArrayLike, correction: ArrayLike) -> NDArray[np.float64]:
    """Positive = improvement over ZERO; exact pathwise identity."""
    y, zero, correction = map(lambda a: np.asarray(a, dtype=float), (y, zero, correction))
    return 2 * (y - zero) * correction - correction ** 2
