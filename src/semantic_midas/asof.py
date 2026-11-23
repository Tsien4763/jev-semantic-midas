"""Publication-aware external-data adapter (data are not included)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .basis import compress_lags, lag_basis


def latest_vintage_lags(frame: pd.DataFrame, origin: str | pd.Timestamp,
                        factors: list[str], n_lags: int = 13) -> tuple[np.ndarray,np.ndarray]:
    """Long-format table: factor, observation_date, release_date, value.

    For each observation date, use the latest version RELEASED by the origin.
    Then select latest n_lags observed dates; irregular elapsed spacing is not
    silently corrected. Callers must supply a regular weekly/monthly grid if
    they need physical calendar lag semantics rather than available slots.
    """
    required={"factor","observation_date","release_date","value"}
    if not required <= set(frame):
        raise ValueError(f"Missing columns: {sorted(required-set(frame))}")
    df=frame.copy()
    for c in ("observation_date","release_date"):
        df[c]=pd.to_datetime(df[c],utc=True,errors="raise")
    time=pd.Timestamp(origin)
    time=time.tz_localize("UTC") if time.tzinfo is None else time.tz_convert("UTC")
    df=df[(df.release_date<=time)&(df.observation_date<=time)]
    df=df.sort_values(["observation_date","release_date"]).drop_duplicates(["factor","observation_date"],keep="last")
    x=np.full((len(factors),n_lags),np.nan)
    for j,f in enumerate(factors):
        v=df[df.factor==f].sort_values("observation_date",ascending=False).value.to_numpy(float)[:n_lags]
        x[j,:len(v)]=v
    return x,np.isfinite(x)


def basis_at_origin(frame: pd.DataFrame, origin, factors, n_lags=13, rank=3):
    x,mask=latest_vintage_lags(frame,origin,factors,n_lags)
    return compress_lags(x,lag_basis(n_lags,rank),mask)
