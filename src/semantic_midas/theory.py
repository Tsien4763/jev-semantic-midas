"""Auditable risk calculations. Guarantees require stated population assumptions."""
from __future__ import annotations
import numpy as np


def forecast_metric(response: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """N,H,d forecast sensitivities -> N,d,d positive-semidefinite exposure metric."""
    t=np.asarray(response,dtype=float)
    if t.ndim!=3 or not np.isfinite(t).all():raise ValueError('Expected finite N,H,d sensitivities')
    w=np.ones(t.shape[1])/t.shape[1] if weights is None else np.asarray(weights,dtype=float)
    if w.shape!=(t.shape[1],) or np.any(w<0) or not np.isfinite(w).all():
        raise ValueError('Weights must be finite, nonnegative and length H')
    return np.einsum('h,nhi,nhj->nij',w,t,t)


def scalar_expected_gain(mu: float,p: float,c: float,exposure: float=1.) -> float:
    if mu<0 or not 0<=p<=1 or c<0 or exposure<0:raise ValueError('Invalid scalar risk arguments')
    return exposure*(2*c*mu*(2*p-1)-c*c)


def robust_scalar_scale(mu_lower: float,p_lower: float,map_error_radius: float=0.) -> float:
    """Conservative scale under UNIFORM conditional reliability/magnitude/map bounds.

    A marginal p confidence interval does not establish the required uniform
    conditional bound when errors depend on forecast exposure.
    """
    if mu_lower<0 or not 0<=p_lower<=1 or map_error_radius<0:raise ValueError('Invalid lower bounds')
    return mu_lower*max(2*p_lower-1,0.)*max(1-map_error_radius,0.)


def bounded_gain_certificate(U: np.ndarray,V: np.ndarray,*,known_U_bound:float,
                             known_V_bound:float,delta:float=.05,cap:float=2.) -> dict:
    """Hoeffding certificate for independent, POPULATION-bounded episode U,V.

    User-supplied support bounds must hold for the population, not merely the
    observed sample. They CANNOT be inferred from empirical maxima. Gaussian
    runs in this repository do NOT satisfy finite support, so this certificate
    is not used or claimed for their results. Clipping would certify a different
    objective unless tail contributions were separately controlled.
    """
    u=np.asarray(U,dtype=float);v=np.asarray(V,dtype=float)
    if u.ndim!=1 or v.shape!=u.shape or len(u)==0:raise ValueError('Paired nonempty episode vectors required')
    if not np.isfinite(u).all() or not np.isfinite(v).all():raise ValueError('Inputs must be finite')
    if known_U_bound<=0 or known_V_bound<=0 or not 0<delta<1 or cap<=0:
        raise ValueError('Invalid population bounds or confidence/cap parameters')
    if np.max(np.abs(u))>known_U_bound+1e-12 or np.min(v)<0 or np.max(v)>known_V_bound+1e-12:
        raise ValueError('Observed values violate the claimed population support')
    n=len(u);l=np.log(4/delta)
    eu=known_U_bound*np.sqrt(2*l/n)
    ev=known_V_bound*np.sqrt(l/(2*n))
    ul=float(u.mean()-eu);vu=float(v.mean()+ev)
    c=float(np.clip(max(ul,0.)/vu,0,cap))
    lower=float(2*c*ul-c*c*vu)
    return {'c':c,'gain_lower':lower,'U_lower':ul,'V_upper':vu,
            'epsilon_U':float(eu),'epsilon_V':float(ev),'delta':delta,
            'assumption':'independent population-bounded calibration episodes'}


def robust_quadratic_entry(U_lower: float, V_upper: float, cap: float = 2.) -> dict:
    """Optimal intervention under a rectangular confidence region.

    If, on an event of probability at least 1-delta, the population alignment
    and energy satisfy U>=U_lower and V<=V_upper, then every c>=0 has
    worst-case gain at least 2*c*U_lower-c^2*V_upper. The maximizer is
    c*=[U_lower]_+/V_upper (subject to cap). Entry is certified only when
    U_lower>0. This theorem is agnostic about how the confidence region was
    constructed.
    """
    if not np.isfinite(U_lower) or not np.isfinite(V_upper) or V_upper <= 0 or cap <= 0:
        raise ValueError('Invalid region/cap')
    c=float(np.clip(max(U_lower,0.)/V_upper,0.,cap))
    lb=float(2*c*U_lower-c*c*V_upper)
    return {'c':c,'gain_lower':lb,'enter':bool(U_lower>0 and c>0),
            'U_lower':float(U_lower),'V_upper':float(V_upper)}


def selective_gain(U: float, V: float, c: float) -> float:
    """Population quadratic gain for any selected/abstained semantic channel."""
    if not all(np.isfinite(x) for x in (U,V,c)) or V < 0 or c < 0:
        raise ValueError('Invalid quadratic-gain inputs')
    return float(2*c*U-c*c*V)
