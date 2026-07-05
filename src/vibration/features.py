import numpy as np
from scipy import stats
from scipy.fft import rfft, rfftfreq
from typing import Dict

def extract_time_features(signal: np.ndarray) -> Dict[str, float]:
    """
    Extracts time-domain features from a 1D vibration signal.
    """
    if len(signal) == 0:
        return {}
        
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    crest_factor = peak / (rms + 1e-9)
    kurtosis = stats.kurtosis(signal, fisher=False)
    skewness = stats.skew(signal)
    variance = np.var(signal)
    
    return {
        "rms": float(rms),
        "peak": float(peak),
        "crest_factor": float(crest_factor),
        "kurtosis": float(kurtosis),
        "skewness": float(skewness),
        "variance": float(variance)
    }

def extract_frequency_features(signal: np.ndarray, sample_rate: float = 20000.0) -> Dict[str, float]:
    """
    Extracts frequency-domain features from a 1D vibration signal using FFT.
    """
    if len(signal) == 0:
        return {}
        
    # Perform Real FFT
    N = len(signal)
    yf = rfft(signal)
    xf = rfftfreq(N, 1 / sample_rate)
    
    # Power spectrum
    power_spectrum = np.abs(yf)**2
    total_power = np.sum(power_spectrum) + 1e-9
    
    # Dominant frequency
    dominant_idx = np.argmax(power_spectrum)
    dominant_freq = xf[dominant_idx]
    
    # Spectral centroid (center of mass of the spectrum)
    spectral_centroid = np.sum(xf * power_spectrum) / total_power
    
    # Spectral entropy (measure of spectral complexity/flatness)
    prob_spectrum = power_spectrum / total_power
    prob_spectrum = prob_spectrum[prob_spectrum > 0] # Avoid log(0)
    spectral_entropy = -np.sum(prob_spectrum * np.log2(prob_spectrum))
    
    return {
        "dominant_freq": float(dominant_freq),
        "spectral_centroid": float(spectral_centroid),
        "spectral_entropy": float(spectral_entropy),
        "total_power": float(total_power)
    }

def extract_all_features(signal: np.ndarray, sample_rate: float = 20000.0) -> Dict[str, float]:
    """
    Extracts all features for machine learning models.
    """
    features = {}
    features.update(extract_time_features(signal))
    features.update(extract_frequency_features(signal, sample_rate))
    return features
