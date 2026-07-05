import os
import requests
import pandas as pd
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger("data_loader")
settings = get_settings()

def load_nab() -> pd.DataFrame:
    """
    Loads the raw NAB CSV file into a pandas DataFrame.
    """
    raw_path = settings.RAW_CSV
    if not os.path.exists(raw_path):
        success = download_data()
        if not success:
            raise FileNotFoundError(f"Failed to obtain raw CSV at {raw_path}")
    df = pd.read_csv(raw_path)
    return df


def download_data() -> bool:
    """
    Downloads the Numenta Anomaly Benchmark (NAB) dataset.
    First, it tries to download using the Kaggle API.
    If Kaggle API is not configured or fails, it falls back to downloading the single CSV directly from GitHub.
    """
    raw_path = settings.RAW_CSV
    if os.path.exists(raw_path):
        logger.info(f"Raw CSV already exists at {raw_path}")
        return True

    # Ensure target directory exists
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)

    # Attempt Kaggle download first
    try:
        logger.info("Attempting to download NAB dataset from Kaggle...")
        import kaggle
        # Authenticate (uses ~/.kaggle/kaggle.json)
        kaggle.api.authenticate()
        # Download boltzmannbrain/nab to data/raw and unzip
        kaggle.api.dataset_download_files("boltzmannbrain/nab", path="data/raw", unzip=True)
        logger.info("Dataset downloaded and unzipped successfully from Kaggle.")
        if os.path.exists(raw_path):
            return True
    except Exception as e:
        logger.warning(f"Kaggle download failed or API token not configured: {e}")

    # Fallback: Download directly from Numenta's GitHub repository
    url = "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/machine_temperature_system_failure.csv"
    try:
        logger.info(f"Attempting fallback download from GitHub: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(raw_path, "wb") as f:
            f.write(response.content)
        logger.info(f"Successfully downloaded raw CSV to {raw_path} via fallback.")
        return True
    except Exception as fallback_e:
        logger.error(f"Fallback download failed: {fallback_e}")
        return False

if __name__ == "__main__":
    success = download_data()
    if success:
        logger.info("Data loader completed successfully.")
    else:
        logger.error("Data loader failed.")
