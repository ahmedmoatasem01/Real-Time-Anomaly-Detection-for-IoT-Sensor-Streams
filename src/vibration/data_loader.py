import os
import pandas as pd
import numpy as np
from typing import List, Tuple
from src.utils.logger import get_logger

logger = get_logger("vibration_loader")

# NASA Bearing Dataset Test 2 specific format
# 20,480 samples per file at 20 kHz
# 4 columns (Bearing 1, Bearing 2, Bearing 3, Bearing 4)
# We will focus on Bearing 1 which fails in Test 2

def load_vibration_snapshot(file_path: str, bearing_idx: int = 0) -> np.ndarray:
    """
    Loads a single 1-second snapshot (20,480 points) for a specific bearing.
    bearing_idx: 0 for Bearing 1, 1 for Bearing 2, etc.
    """
    try:
        # The files are tab-separated, no headers
        df = pd.read_csv(file_path, sep='\t', header=None)
        # Select the specific bearing column and return as numpy array
        return df[bearing_idx].values
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return np.array([])

def get_snapshot_files(data_dir: str) -> List[str]:
    """
    Returns a chronologically sorted list of snapshot file paths in the directory.
    """
    if not os.path.exists(data_dir):
        logger.error(f"Directory {data_dir} does not exist.")
        return []
        
    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    # The file names are timestamps like '2004.02.12.10.32.39'
    files.sort()
    return [os.path.join(data_dir, f) for f in files]
