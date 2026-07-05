import os
import time
import json
import joblib
import pandas as pd
import numpy as np
from collections import deque
from typing import Dict, Any, Tuple
from src.utils.config import get_settings
from src.utils.logger import get_logger
from src.features.feature_engineering import make_features
from src.drift.drift_service import drift_service
from src.synthetic.synthetic_stream import fault_manager
from src.models.train_river_online import RiverHSTWrapper
import __main__
# Hack to allow uvicorn to unpickle a class defined in a script run as __main__
__main__.RiverHSTWrapper = RiverHSTWrapper

logger = get_logger("inference_service")
settings = get_settings()

class InferenceService:
    def __init__(self):
        self.model_dir = settings.MODEL_DIR
        self.scaler = None
        self.model = None
        self.threshold = 0.5
        self.feature_columns = []
        self.feature_stats = {}
        
        # Deque buffer per sensor. Maxlen 65 to ensure enough lag and rolling window 60
        self.buffers: Dict[str, deque] = {}
        self.max_buffer_len = 65
        
        self.load_artifacts()
        self.preload_buffers()

    def load_artifacts(self):
        try:
            scaler_path = os.path.join(self.model_dir, "scaler.pkl")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Loaded scaler from {scaler_path}")
                
            model_path = os.path.join(self.model_dir, "isolation_forest.pkl")
            # For ensemble, we can load all available models
            self.ensemble_models = {}
            for m_name in ["isolation_forest", "elliptic_envelope", "one_class_svm", "lof", "river_online"]:
                m_path = os.path.join(self.model_dir, f"{m_name}.pkl")
                if os.path.exists(m_path):
                    self.ensemble_models[m_name] = joblib.load(m_path)
                    logger.info(f"Loaded ensemble model {m_name}")
            
            # The primary model from registry
            from src.registry.model_registry import list_models
            registry_list = list_models()
            prod_entry = next((entry for entry in registry_list if entry.get("is_production")), {})
            prod_model_name = prod_entry.get("name", "isolation_forest")
            
            artifact_path = prod_entry.get("artifact_path")
            
            if artifact_path and os.path.exists(artifact_path):
                self.model = joblib.load(artifact_path)
            elif os.path.exists(model_path):
                self.model = self.ensemble_models.get("isolation_forest")
                
            # Wait, if we use set_production in main.py, it updates self.model.
            # But we also need self.model_name
            self.model_name = prod_model_name
            
            # Use threshold from registry
            threshold_path = prod_entry.get("threshold_path", os.path.join(self.model_dir, "threshold.json"))
            self.ensemble_thresholds = {}
            for m_name in ["isolation_forest", "elliptic_envelope", "one_class_svm", "lof", "river_online"]:
                # the threshold files are named threshold_<name>.json usually, except isolation_forest is threshold.json
                if m_name == "isolation_forest":
                    t_path = os.path.join(self.model_dir, "threshold.json")
                elif m_name == "elliptic_envelope":
                    t_path = os.path.join(self.model_dir, "threshold_elliptic.json")
                elif m_name == "one_class_svm":
                    t_path = os.path.join(self.model_dir, "threshold_ocsvm.json")
                else:
                    t_path = os.path.join(self.model_dir, f"threshold_{m_name}.json")
                    
                if os.path.exists(t_path):
                    with open(t_path, "r") as f:
                        self.ensemble_thresholds[m_name] = json.load(f)["threshold"]
                        
            if os.path.exists(threshold_path):
                with open(threshold_path, "r") as f:
                    t_data = json.load(f)
                self.threshold = t_data["threshold"]
                logger.info(f"Loaded threshold {self.threshold:.6f} from {threshold_path}")
                
            feature_cols_path = os.path.join(self.model_dir, "feature_columns.json")
            if os.path.exists(feature_cols_path):
                with open(feature_cols_path, "r") as f:
                    self.feature_columns = json.load(f)
                logger.info(f"Loaded feature columns from {feature_cols_path}")
                
            feature_stats_path = os.path.join(self.model_dir, "feature_stats.json")
            if os.path.exists(feature_stats_path):
                with open(feature_stats_path, "r") as f:
                    self.feature_stats = json.load(f)
                logger.info(f"Loaded feature stats from {feature_stats_path}")
        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")

    def preload_buffers(self):
        """
        Pre-loads the rolling buffer with historical training data to avoid warm-up issues
        """
        if os.path.exists(settings.PROCESSED_CSV):
            try:
                df = pd.read_csv(settings.PROCESSED_CSV)
                train_df = df[df["split"] == "train"].sort_values("timestamp")  # type: ignore[call-overload]
                if len(train_df) > 0:
                    sensor_id = settings.SENSOR_ID
                    self.buffers[sensor_id] = deque(maxlen=self.max_buffer_len)
                    last_rows = train_df.tail(self.max_buffer_len)
                    for _, row in last_rows.iterrows():
                        self.buffers[sensor_id].append((row["timestamp"], row["value_scaled"]))
                    logger.info(f"Preloaded buffer for sensor '{sensor_id}' with {len(self.buffers[sensor_id])} historical readings.")
            except Exception as e:
                logger.warning(f"Could not preload buffers from processed CSV: {e}")

    def get_buffer(self, sensor_id: str) -> deque:
        if sensor_id not in self.buffers:
            self.buffers[sensor_id] = deque(maxlen=self.max_buffer_len)
        return self.buffers[sensor_id]

    def explain_anomaly(self, feature_vector: Dict[str, float]) -> str:
        """
        Explains which feature contributed most to the anomaly using deviations from training mean,
        and appends a rule-based maintenance recommendation.
        """
        if not self.feature_stats:
            return "No explanation available (feature stats missing)."
            
        max_deviation = -1.0
        contributing_feature = ""
        contributing_val = 0.0
        contributing_mean = 0.0
        
        for feature, val in feature_vector.items():
            if feature in self.feature_stats:
                mean = self.feature_stats[feature]["mean"]
                std = self.feature_stats[feature]["std"]
                deviation = abs(val - mean) / (std + 1e-9)
                if deviation > max_deviation:
                    max_deviation = deviation
                    contributing_feature = feature
                    contributing_val = val
                    contributing_mean = mean
                    
        reason = ""
        if contributing_feature:
            reason = (f"'{contributing_feature}' is {max_deviation:.1f} std devs from baseline "
                      f"({contributing_val:.3f} vs normal {contributing_mean:.3f}).")
        else:
            reason = "Unknown reason (all features are within normal deviation)."

        # Append rule-based recommendation
        action = "monitor system"
        if "value" in contributing_feature or "Z-score" in reason:
            action = "inspect cooling / thermal sensors"
        elif "rate" in contributing_feature or "diff" in contributing_feature:
            action = "check for sudden mechanical shocks or sensor dropouts"
        elif "RMS" in contributing_feature or "kurtosis" in contributing_feature:
            action = "inspect bearing for degradation"
            
        return f"{reason} Rec: {action}."

    def predict(self, timestamp: str, sensor_id: str, value: float) -> Dict[str, Any]:
        """
        Predicts whether a single reading is anomalous and determines severity + reason.
        """
        start_time = time.perf_counter()

        # 0. Apply synthetic faults if any are active
        injected_value = fault_manager.apply(value)
        
        if injected_value is None:
            # For missing values, we return early with an error or handle it.
            # In our current pipeline, missing values could be skipped or passed.
            # Since the spec says "match whatever preprocessing does for gaps", and 
            # currently inference_service expects a float, we can just return a non-anomaly.
            inference_ms = (time.perf_counter() - start_time) * 1000
            return {
                "timestamp": timestamp,
                "sensor_id": sensor_id,
                "value": value,
                "anomaly_score": 0.0,
                "is_anomaly": False,
                "severity": "none",
                "reason": "Missing value injected",
                "model": "none",
                "inference_ms": inference_ms,
            }
        
        value = injected_value

        # 1. Scale value — pass as single-column DataFrame so sklearn sees the
        #    exact feature name it was fit on ('value'), eliminating the UserWarning.
        if self.scaler is not None:
            value_df = pd.DataFrame([[value]], columns=["value"])  # type: ignore[list-item]
            value_scaled = float(self.scaler.transform(value_df)[0][0])
        else:
            value_scaled = value

        # 2. Update rolling buffer
        buffer = self.get_buffer(sensor_id)
        buffer.append((timestamp, value_scaled))

        # Guard: model or feature contract not loaded yet
        if self.model is None or not self.feature_columns:
            inference_ms = (time.perf_counter() - start_time) * 1000
            return {
                "timestamp": timestamp,
                "sensor_id": sensor_id,
                "value": value,
                "anomaly_score": 0.0,
                "is_anomaly": False,
                "severity": "none",
                "reason": "Model or features not loaded",
                "model": "none",
                "inference_ms": inference_ms,
            }

        # 3. Build feature matrix from buffer
        scaled_values = [item[1] for item in list(buffer)]
        temp_df = pd.DataFrame({"value_scaled": scaled_values})
        temp_features = make_features(temp_df)
        last_row = temp_features.iloc[-1]

        # Assemble feature dict in canonical column order
        feature_dict = {}
        for col in self.feature_columns:
            val = last_row.get(col, 0.0)
            feature_dict[col] = 0.0 if pd.isna(val) else float(val)

        # 4. Score — pass as DataFrame so sklearn sees column names (no UserWarning)
        X_df = pd.DataFrame([feature_dict], columns=self.feature_columns)  # type: ignore[list-item]
        score = float(-self.model.score_samples(X_df)[0])
        
        # 5. Check anomaly and set severity
        is_anomaly = score > self.threshold
        severity = "none"
        reason = "Normal operation"
        
        if is_anomaly:
            diff = score - self.threshold
            if diff < 1.0:
                severity = "low"
            elif diff < 3.0:
                severity = "medium"
            else:
                severity = "high"
                
            reason = self.explain_anomaly(feature_dict)
                
            
        # Update drift detector
        drift_service.update(feature_dict)
            
        inference_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            "timestamp": timestamp,
            "sensor_id": sensor_id,
            "value": value,
            "anomaly_score": score,
            "is_anomaly": is_anomaly,
            "severity": severity,
            "reason": reason,
            "model": "isolation_forest",
            "inference_ms": inference_ms
        }

    def ensemble_predict(self, timestamp: str, sensor_id: str, value: float) -> Dict[str, Any]:
        """
        Scores the reading across all available models to provide a consensus anomaly decision.
        """
        start_time = time.perf_counter()
        
        # 1. Scale value
        if self.scaler is not None:
            value_df = pd.DataFrame([[value]], columns=["value"])
            value_scaled = float(self.scaler.transform(value_df)[0][0])
        else:
            value_scaled = value

        # 2. Extract features from current buffer (which was updated by predict() typically, 
        # but to be safe we don't double-append if this is called separately).
        # We will assume ensemble_predict is called AFTER predict() updates the buffer, OR 
        # if called standalone, we just use the current buffer.
        buffer = self.get_buffer(sensor_id)
        if len(buffer) == 0:
            return {"error": "Buffer empty"}
            
        scaled_values = [item[1] for item in list(buffer)]
        temp_df = pd.DataFrame({"value_scaled": scaled_values})
        temp_features = make_features(temp_df)
        last_row = temp_features.iloc[-1]

        feature_dict = {}
        for col in self.feature_columns:
            val = last_row.get(col, 0.0)
            feature_dict[col] = 0.0 if pd.isna(val) else float(val)

        X_df = pd.DataFrame([feature_dict], columns=self.feature_columns)
        
        votes = 0
        total_models = len(self.ensemble_models)
        details = {}
        
        for m_name, m_obj in self.ensemble_models.items():
            try:
                if m_name == "river_online":
                    # River Wrapper uses score_samples which returns negated scores
                    score = float(-m_obj.score_samples(X_df)[0])
                elif hasattr(m_obj, "score_samples"):
                    score = float(-m_obj.score_samples(X_df)[0])
                else:
                    score = float(-m_obj.decision_function(X_df)[0])
                    
                thresh = self.ensemble_thresholds.get(m_name, 0.5)
                is_anom = bool(score > thresh)
                
                details[m_name] = {
                    "score": score,
                    "threshold": thresh,
                    "is_anomaly": is_anom
                }
                
                if is_anom:
                    votes += 1
            except Exception as e:
                logger.error(f"Ensemble error for {m_name}: {e}")
                
        confidence = (votes / total_models) if total_models > 0 else 0.0
        
        inference_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            "timestamp": timestamp,
            "sensor_id": sensor_id,
            "value": value,
            "votes": votes,
            "total_models": total_models,
            "ensemble_confidence": confidence,
            "is_anomaly": votes >= (total_models / 2) if total_models > 0 else False,
            "message": f"{votes} of {total_models} agree \u2192 anomaly, {confidence*100:.0f}%",
            "model_details": details,
            "inference_ms": inference_ms
        }
