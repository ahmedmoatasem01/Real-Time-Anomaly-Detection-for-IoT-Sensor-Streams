import torch
import torch.nn as nn
import numpy as np

class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len: int, n_features: int, hidden_dim: int = 64):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        
        self.encoder = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, n_features, batch_first=True)
        
    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        hidden = hidden[-1]
        hidden = hidden.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.decoder(hidden)
        return out

class LSTMAutoencoderWrapper:
    """
    Wrapper for LSTM Autoencoder to expose a scikit-learn like interface
    so it can be dropped into evaluate_all.py and inference_service.py.
    """
    def __init__(self, seq_len: int, n_features: int, hidden_dim: int = 64, device: str = "cpu"):
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.device = device
        self.model = LSTMAutoencoder(seq_len, n_features, hidden_dim).to(device)
        self.model.eval()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _create_sequences(self, X: np.ndarray) -> np.ndarray:
        # Create sliding windows
        n_samples = len(X)
        if n_samples < self.seq_len:
            # Pad if too short
            pad = np.zeros((self.seq_len - n_samples, self.n_features))
            X = np.vstack([pad, X])
            n_samples = len(X)
            
        seqs = []
        for i in range(n_samples - self.seq_len + 1):
            seqs.append(X[i:i+self.seq_len])
        return np.array(seqs)

    def score_samples(self, X) -> np.ndarray:
        """
        Returns -MSE anomaly score so that smaller values mean more anomalous (like IsolationForest).
        Actually, sklearn models return negative anomaly scores, where lower means more anomalous.
        High MSE -> More anomalous. So return -MSE.
        """
        if isinstance(X, list):
            X = np.array(X)
        elif hasattr(X, "values"):
            X = X.values
            
        # Create sequences
        # Output shape should match X length.
        # For the first (seq_len - 1) items, we pad or repeat the first value to align output length.
        
        # Pad beginning
        pad = np.repeat([X[0]], self.seq_len - 1, axis=0)
        X_padded = np.vstack([pad, X])
        
        seqs = self._create_sequences(X_padded)
        seqs_tensor = torch.tensor(seqs, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            reconstructions = self.model(seqs_tensor)
            # MSE of the last step in the sequence
            mse = torch.mean((reconstructions[:, -1, :] - seqs_tensor[:, -1, :]) ** 2, dim=1)
            
        # Return negative MSE
        return -mse.cpu().numpy()
