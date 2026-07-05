from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    RAW_CSV: str = "data/raw/realKnownCause/machine_temperature_system_failure.csv"
    PROCESSED_CSV: str = "data/processed/nab_processed.csv"
    DB_URL: str = "sqlite:///reports/anomaly.db"
    API_URL: str = "http://localhost:8000"
    MODEL_DIR: str = "models"
    SENSOR_ID: str = "machine_temperature"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

@lru_cache()
def get_settings() -> Settings:
    return Settings()
