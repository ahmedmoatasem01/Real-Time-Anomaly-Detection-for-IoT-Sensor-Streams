import os
import time
import argparse
import requests
import pandas as pd
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger("stream_simulator")
settings = get_settings()

def run_simulator(csv_path: str, speed: float, loop: bool, start_index: int, jump_to_anomaly: int = 0):
    logger.info(f"Starting stream simulator on {csv_path}...")
    logger.info(f"Configuration: speed={speed}x, loop={loop}, start_index={start_index}, jump_to_anomaly={jump_to_anomaly}")
    
    if not os.path.exists(csv_path):
        if os.path.exists(settings.PROCESSED_CSV):
            csv_path = settings.PROCESSED_CSV
            logger.warning(f"Specified CSV not found, using fallback processed CSV: {csv_path}")
        else:
            raise FileNotFoundError(f"CSV file not found at {csv_path}")
            
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    n = len(df)
    logger.info(f"Loaded {n} rows from CSV.")
    
    idx = start_index
    
    if jump_to_anomaly > 0:
        # Find the N-th anomaly window
        is_anomaly = df["label"].values
        in_window = False
        window_starts = []
        for i in range(len(is_anomaly)):
            if is_anomaly[i] == 1 and not in_window:
                window_starts.append(i)
                in_window = True
            elif is_anomaly[i] == 0:
                in_window = False
                
        if jump_to_anomaly <= len(window_starts):
            # Jump to 50 steps before the anomaly starts (about 4 hours)
            target = window_starts[jump_to_anomaly - 1]
            idx = max(0, target - 50)
            logger.info(f"Jumping to anomaly #{jump_to_anomaly} at index {target}. Starting stream at {idx}")
        else:
            logger.warning(f"Requested anomaly #{jump_to_anomaly} but only found {len(window_starts)} windows.")
            
    if idx >= n or idx < 0:
        logger.warning(f"Start index {idx} out of bounds, resetting to 0.")
        idx = 0
        
    predict_url = f"{settings.API_URL}/predict"
    logger.info(f"Target API Endpoint: {predict_url}")
    
    base_cadence_seconds = 300.0
    sleep_seconds = base_cadence_seconds / speed
    logger.info(f"Step interval sleep: {sleep_seconds:.4f} seconds")
    
    while True:
        row = df.iloc[idx]
        timestamp_str = row["timestamp"].strftime("%Y-%m-%dT%H:%M:%S")
        
        payload = {
            "timestamp": timestamp_str,
            "sensor_id": str(row["sensor_id"]),
            "value": float(row["value"])
        }
        
        success = False
        retries = 3
        backoff = 1.0
        
        for attempt in range(retries):
            try:
                response = requests.post(predict_url, json=payload, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    ts = data["timestamp"]
                    val = data["value"]
                    score = data["anomaly_score"]
                    is_anom = data["is_anomaly"]
                    sev = data["severity"]
                    
                    status_indicator = "ANOMALY" if is_anom else "NORMAL "
                    logger.info(
                        f"[{status_indicator}] {ts} | Val: {val:6.2f} | Score: {score:.4f} | Sev: {sev:6s} | Reason: {data['reason']}"
                    )
                    success = True
                    break
                else:
                    logger.warning(f"API returned status {response.status_code} on attempt {attempt+1}")
            except Exception as e:
                logger.warning(f"Connection error to API on attempt {attempt+1}: {e}")
                
            time.sleep(backoff)
            backoff *= 2.0
            
        if not success:
            logger.error(f"Failed to send reading at index {idx} after {retries} attempts. Skipping.")
            
        idx += 1
        if idx >= n:
            if loop:
                logger.info("Reached end of stream, looping back to start_index.")
                idx = start_index
            else:
                logger.info("Reached end of stream, terminating simulator.")
                break
                
        time.sleep(sleep_seconds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IoT Sensor Stream Simulator")
    parser.add_argument("--csv", type=str, default="data/sample_stream/test_stream.csv", help="CSV file path to replay")
    parser.add_argument("--speed", type=float, default=50.0, help="Replay speed multiplier (e.g. 50)")
    parser.add_argument("--loop", action="store_true", help="Loop the stream upon reaching the end")
    parser.add_argument("--start-index", type=int, default=0, help="Index inside CSV to start replaying from")
    parser.add_argument("--jump-to-anomaly", type=int, default=0, help="Jump to N-th anomaly (e.g. 1 for the first anomaly). Offsets by 50 steps before it.")
    
    args = parser.parse_args()
    
    try:
        run_simulator(
            csv_path=args.csv,
            speed=args.speed,
            loop=args.loop,
            start_index=args.start_index,
            jump_to_anomaly=args.jump_to_anomaly
        )
    except KeyboardInterrupt:
        logger.info("Simulator stopped by user.")
    except Exception as exc:
        logger.critical(f"Simulator crashed: {exc}")
