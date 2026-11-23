"""CPU-only semantic regime priors for linear-basis mixed-frequency filters."""
from .basis import lag_basis, compress_lags, structural_map
from .filter import FilterConfig, FilterResult, same_filter_paths
from .calibration import fit_scale, calibrate, forecast_arms
__version__ = "1.0.0"
