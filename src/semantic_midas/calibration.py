"""Magnitude comes from disjoint historical outcomes, not test truth.

CALIBRATED needs historic signs and numeric outcomes but no true break
magnitudes or true signs. P_SHRINK additionally assumes audited historical
correctness labels. GATE is a heuristic held-out empirical-risk check.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from scipy.stats import beta


def direction_response(response: np.ndarray, G: np.ndarray, signs: np.ndarray) -> np.ndarray:
    if response.shape[0] != signs.shape[0] or G.shape[1] != signs.shape[1]:
        raise ValueError("Response/sign dimensions do not match")
    v = signs @ G.T
    return np.einsum("ntd,nd->nt", response, v)


def fit_scale(residual: np.ndarray, unit_correction: np.ndarray,
              cap: float = 2.) -> tuple[float, float, float]:
    """Minimize mean (r - c*b)^2 over 0<=c<=cap on historical data only."""
    if residual.shape != unit_correction.shape or not np.isfinite(residual).all():
        raise ValueError("Finite, matched residual and correction arrays required")
    U = float(np.mean(residual * unit_correction))
    V = float(np.mean(unit_correction ** 2))
    c = 0. if V <= 1e-15 else float(np.clip(U / V, 0., cap))
    return c, U, V


def cp_lower(correct: int, total: int, alpha: float = .05) -> float:
    """One-sided Clopper-Pearson lower bound for iid audited Bernoulli signs."""
    if not 0 <= correct <= total or total <= 0:
        raise ValueError("Invalid binomial counts")
    return 0. if correct == 0 else float(beta.ppf(alpha, correct, total - correct + 1))


def bootstrap_lower(values: np.ndarray, rng: np.random.Generator,
                    reps: int = 2000, alpha: float = .05) -> float:
    """Episode-level percentile-bootstrap bound, NOT a universal certificate."""
    z = np.asarray(values, dtype=float)
    if z.ndim != 1 or not len(z) or not np.isfinite(z).all():
        raise ValueError("Expected a nonempty finite episode vector")
    idx = rng.integers(0, len(z), (reps, len(z)))
    return float(np.quantile(z[idx].mean(axis=1), alpha))


@dataclass(frozen=True)
class Calibration:
    raw_scale: float
    calibrated_scale: float
    p_hat: float
    p_lower_iid: float
    p_shrink_scale: float
    gate_scale: float
    gate_gain_lower: float
    gate_mean_gain: float
    fit_U: float
    fit_V: float
    n_fit_episodes: int
    n_gate_episodes: int
    label: str

    def as_dict(self) -> dict:
        return asdict(self)


def calibrate(fit_r: np.ndarray, fit_b: np.ndarray, fit_truth_b: np.ndarray,
              fit_a: np.ndarray, fit_truth_sign: np.ndarray,
              gate_r: np.ndarray, gate_b: np.ndarray,
              *, cap: float = 2., reps: int = 2000, seed: int = 415,
              label: str = "matched") -> Calibration:
    # The raw-scale benchmark uses audited correct historic directions.
    raw, _, _ = fit_scale(fit_r, fit_truth_b, cap)
    c, U, V = fit_scale(fit_r, fit_b, cap)
    nonzero = fit_a != 0
    total = int(nonzero.sum())
    correct = int(((fit_a == fit_truth_sign) & nonzero).sum())
    p_hat = correct / total if total else .5
    p_lower = cp_lower(correct, total) if total else 0.
    pscale = raw * max(2 * p_hat - 1, 0.)
    gain = (2 * gate_r * (c * gate_b) - (c * gate_b)**2).mean(axis=1)
    lower = bootstrap_lower(gain, np.random.default_rng(seed), reps)
    return Calibration(raw, c, p_hat, p_lower, pscale, c if lower > 0 else 0.,
                       lower, float(gain.mean()), U, V,
                       len(fit_r), len(gate_r), label)


def forecast_arms(zero: np.ndarray, response: np.ndarray, G: np.ndarray,
                  signs: np.ndarray, calib: Calibration,
                  *, true_delta: np.ndarray | None = None,
                  shuffle_seed: int = 551) -> dict[str, np.ndarray]:
    """Predict from frozen scales. true_delta is quarantined to ORACLE only."""
    b = direction_response(response, G, signs)
    perm = np.random.default_rng(shuffle_seed).permutation(len(signs))
    shuffled_b = direction_response(response, G, signs[perm])
    out = {
        "ZERO": zero.copy(),
        "RAW": zero + calib.raw_scale * b,
        "P_SHRINK": zero + calib.p_shrink_scale * b,
        "CALIBRATED": zero + calib.calibrated_scale * b,
        "GATE": zero + calib.gate_scale * b,
        "OPPOSITE": zero - calib.calibrated_scale * b,
        "SHUFFLED": zero + calib.calibrated_scale * shuffled_b,
    }
    if true_delta is not None:
        out["ORACLE"] = zero + np.einsum("ntd,nd->nt", response, true_delta)
    return out
