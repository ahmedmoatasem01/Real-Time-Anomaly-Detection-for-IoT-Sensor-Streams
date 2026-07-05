import math
import numpy as np
from scipy.stats import norm

def population_stability_index(expected_pct, actual_pct, eps=1e-6) -> float:
    """
    PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))
    """
    psi = 0.0
    for a, e in zip(actual_pct, expected_pct):
        a_safe = max(a, eps)
        e_safe = max(e, eps)
        psi += (a_safe - e_safe) * math.log(a_safe / e_safe)
    return psi

def mean_shift_sigma(baseline_mean: float, baseline_std: float, live_mean: float) -> float:
    return (live_mean - baseline_mean) / (baseline_std + 1e-9)

def classify(psi: float, mean_shift: float) -> str:
    """
    stable  : psi < 0.1  and |shift| < 2
    warning : 0.1 <= psi < 0.25 or 2 <= |shift| < 3
    critical: psi >= 0.25 or |shift| >= 3
    """
    abs_shift = abs(mean_shift)
    if psi >= 0.25 or abs_shift >= 3.0:
        return "critical"
    elif psi >= 0.1 or abs_shift >= 2.0:
        return "warning"
    else:
        return "stable"

def compute_drift_for_feature(live_values: list, baseline_mean: float, baseline_std: float) -> tuple:
    if not live_values:
        return 0.0, 0.0
        
    live_mean = np.mean(live_values)
    shift = mean_shift_sigma(baseline_mean, baseline_std, live_mean)
    
    # Bucketize for PSI
    # We create 10 bins between mean - 3*std and mean + 3*std
    bins = np.linspace(baseline_mean - 3 * baseline_std, baseline_mean + 3 * baseline_std, 11)
    
    # Expected percentages from Normal distribution
    expected_pct = []
    for i in range(10):
        # cdf(upper) - cdf(lower)
        if i == 0:
            p = norm.cdf(bins[1], loc=baseline_mean, scale=baseline_std)
        elif i == 9:
            p = 1.0 - norm.cdf(bins[9], loc=baseline_mean, scale=baseline_std)
        else:
            p = norm.cdf(bins[i+1], loc=baseline_mean, scale=baseline_std) - norm.cdf(bins[i], loc=baseline_mean, scale=baseline_std)
        expected_pct.append(p)
        
    # Actual percentages
    actual_counts, _ = np.histogram(live_values, bins=bins)
    # Add out of bounds to first and last bins
    out_left = sum(1 for v in live_values if v < bins[0])
    out_right = sum(1 for v in live_values if v > bins[-1])
    
    actual_counts[0] += out_left
    actual_counts[-1] += out_right
    
    actual_pct = actual_counts / len(live_values)
    
    psi = population_stability_index(expected_pct, actual_pct)
    
    return float(psi), float(shift)
