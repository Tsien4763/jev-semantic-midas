"""Synthetic demo for Jev-Semantic MIDAS calibration & gating core (S42203).

NOTE: Uses synthetic multi-episode residuals and noisy signs only — NOT a
reproduction of the 60-quarter FOMC inflation nowcasting experiment.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from semantic_midas.calibration import calibrate, direction_response, forecast_arms  # noqa: E402


def simulate_episodes(rng: np.random.Generator, n: int, horizon: int = 4, acc: float = 0.80):
    G = np.array([[1.0]])
    response = np.ones((n, horizon, 1)) * np.linspace(1.0, 0.5, horizon)[None, :, None]
    true_sign = rng.choice([-1.0, 1.0], size=(n, 1))
    flip = rng.random((n, 1)) > acc
    obs_sign = np.where(flip, -true_sign, true_sign)
    true_delta = 0.8 * true_sign
    truth_b = direction_response(response, G, true_sign)
    obs_b = direction_response(response, G, obs_sign)
    resid = 0.8 * truth_b + rng.normal(0.0, 0.35, size=(n, horizon))
    return G, response, true_sign, obs_sign, true_delta, truth_b, obs_b, resid


def main() -> None:
    print("=== Jev-Semantic MIDAS MVP (synthetic demo) ===")
    rng = np.random.default_rng(42)

    G, _, fit_ts, fit_os, _, fit_tb, fit_ob, fit_r = simulate_episodes(rng, n=60, acc=0.82)
    _, _, _, _, _, _, gate_ob, gate_r = simulate_episodes(rng, n=40, acc=0.82)
    _, te_resp, _, te_os, te_td, _, _, te_r = simulate_episodes(rng, n=50, acc=0.82)

    calib = calibrate(
        fit_r=fit_r,
        fit_b=fit_ob,
        fit_truth_b=fit_tb,
        fit_a=fit_os[:, 0],
        fit_truth_sign=fit_ts[:, 0],
        gate_r=gate_r,
        gate_b=gate_ob,
        reps=500,
        seed=42,
    )
    print(
        f"[1/2] Frozen calibration: c_hat={calib.calibrated_scale:.3f} | p_hat={calib.p_hat:.2%} "
        f"(CP lower={calib.p_lower_iid:.2%}) | gate_scale={calib.gate_scale:.3f} (gate_lower={calib.gate_gain_lower:+.4f})"
    )

    zero_pred = np.zeros_like(te_r)
    arms = forecast_arms(zero_pred, te_resp, G, te_os, calib, true_delta=te_td, shuffle_seed=99)
    print("[2/2] Synthetic test MSE by arm (lower is better):")
    for name in ("ZERO", "CALIBRATED", "GATE", "P_SHRINK", "SHUFFLED", "OPPOSITE", "ORACLE"):
        mse = float(np.mean((te_r - arms[name]) ** 2))
        print(f"      {name:11s}: MSE={mse:.4f}")


if __name__ == "__main__":
    main()
