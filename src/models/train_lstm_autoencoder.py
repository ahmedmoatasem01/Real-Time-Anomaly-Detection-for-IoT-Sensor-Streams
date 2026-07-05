import os
import json
import torch
import joblib
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support

from src.models.lstm_wrapper import LSTMAutoencoderWrapper, LSTMAutoencoder
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger("train_lstm")
settings = get_settings()

def _create_sequences(X: np.ndarray, seq_len: int) -> np.ndarray:
    n_samples = len(X)
    if n_samples < seq_len:
        pad = np.zeros((seq_len - n_samples, X.shape[1]))
        X = np.vstack([pad, X])
        n_samples = len(X)
        
    seqs = []
    for i in range(n_samples - seq_len + 1):
        seqs.append(X[i:i+seq_len])
    return np.array(seqs)

def train_lstm():
    logger.info("Starting LSTM Autoencoder training...")
    
    df = pd.read_csv(settings.PROCESSED_CSV)
    
    feature_cols_path = os.path.join(settings.MODEL_DIR, "feature_columns.json")
    with open(feature_cols_path, "r") as f:
        feature_cols = json.load(f)
        
    logger.info(f"Using {len(feature_cols)} feature columns from feature_columns.json")
    
    # We only use 'normal' data for training an autoencoder usually.
    # The processed_data has label=1 for anomalies.
    train_clean = df[(df["split"] == "train") & (df["label"] == 0)].dropna(subset=feature_cols)  # type: ignore[call-overload]
    X_train = np.asarray(train_clean[feature_cols].values)
    
    val_clean = df[df["split"] == "val"].dropna(subset=feature_cols)  # type: ignore[call-overload]
    X_val = val_clean[feature_cols].values
    y_val = val_clean["label"].values
    
    SEQ_LEN = 15
    HIDDEN_DIM = 64
    BATCH_SIZE = 128
    EPOCHS = 10
    LR = 1e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Device: {DEVICE}, seq_len={SEQ_LEN}, hidden_dim={HIDDEN_DIM}, epochs={EPOCHS}")
    
    train_seqs = _create_sequences(X_train, SEQ_LEN)
    train_tensor = torch.tensor(train_seqs, dtype=torch.float32)
    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = LSTMAutoencoder(SEQ_LEN, len(feature_cols), HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in train_loader:
            x_batch = batch[0].to(DEVICE)
            optimizer.zero_grad()
            reconstructed = model(x_batch)
            loss = criterion(reconstructed, x_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)
            
        avg_loss = total_loss / len(train_dataset)
        logger.info(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_loss:.6f}")
        
    # Save the model state dict
    os.makedirs(settings.MODEL_DIR, exist_ok=True)
    model_path = os.path.join(settings.MODEL_DIR, "lstm_autoencoder.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved LSTM Autoencoder weights to {model_path}")
    
    # Also save the wrapper using joblib for evaluate_all compatibility
    wrapper = LSTMAutoencoderWrapper(SEQ_LEN, len(feature_cols), HIDDEN_DIM, DEVICE)
    wrapper.load_state_dict(model.state_dict())
    wrapper_path = os.path.join(settings.MODEL_DIR, "lstm_autoencoder.pkl")
    joblib.dump(wrapper, wrapper_path)
    logger.info(f"Saved LSTM Wrapper to {wrapper_path}")
    
    # Tune threshold on validation set
    logger.info("Tuning threshold on validation set...")
    val_scores = wrapper.score_samples(X_val) # returns negative MSE
    
    # Test percentiles to find the best F1
    best_thresh = None
    best_f1 = 0
    best_prec = 0
    best_rec = 0
    
    # Because scores are negative MSE, anomalies have smaller (more negative) values
    percentiles = np.linspace(0.1, 10.0, 50)
    for p in percentiles:
        thresh = np.percentile(val_scores, p)
        preds = (val_scores < thresh).astype(int)
        
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_val, preds, average="binary", zero_division=0.0  # type: ignore[call-overload]
        )
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_prec = prec
            best_rec = rec
            
    # Save threshold
    if best_thresh is None:
        best_thresh = float(np.percentile(val_scores, 1.0))
        
    threshold_data = {
        "threshold": float(best_thresh),
        "split": "validation",
        "selection_metrics": {
            "Precision": float(best_prec),
            "Recall": float(best_rec),
            "F1": float(best_f1)
        },
        "notes": f"LSTM Autoencoder, seq_len={SEQ_LEN}, hidden_dim={HIDDEN_DIM}"
    }
    
    thresh_path = os.path.join(settings.MODEL_DIR, "threshold_lstm.json")
    with open(thresh_path, "w") as f:
        json.dump(threshold_data, f, indent=4)
        
    logger.info(f"Selected threshold={best_thresh:.6f}")
    logger.info(f"Validation metrics: Precision={best_prec:.4f}, Recall={best_rec:.4f}, F1={best_f1:.4f}")
    logger.info(f"Saved threshold configuration to {thresh_path}")

if __name__ == "__main__":
    train_lstm()
