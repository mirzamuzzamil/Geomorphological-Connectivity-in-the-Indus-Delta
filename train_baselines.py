# /// script
# dependencies = [
#   "scikit-learn",
#   "torch",
#   "numpy",
#   "scipy"
# ]
# ///

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score

# ----------------------------------------------------
# BASELINE A: Random Forest (Spatial/Direct Classifier)
# ----------------------------------------------------
class RandomForestBaseline:
    def __init__(self, max_depth=8, n_estimators=100):
        self.model = RandomForestClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        
    def fit(self, X_train, y_train):
        print("Training Random Forest Baseline A...")
        # X_train shape: (num_samples, num_features)
        # y_train shape: (num_samples,)
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
        
    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        print("\n--- Random Forest Baseline A Evaluation ---")
        print(classification_report(y_test, preds))
        return f1_score(y_test, preds)

# ----------------------------------------------------
# BASELINE B: PyTorch LSTM (Temporal-only Sequence)
# ----------------------------------------------------
class LSTMBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, output_dim=1):
        super(LSTMBaseline, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Input shape x: (batch_size, sequence_length, input_dim)
        lstm_out, _ = self.lstm(x) # shape: (batch_size, sequence_length, hidden_dim)
        out = self.fc(lstm_out)   # shape: (batch_size, sequence_length, output_dim)
        return self.sigmoid(out)

# ----------------------------------------------------
# BASELINE C: PyTorch Transformer (Spatio-Temporal Attention)
# ----------------------------------------------------
class TransformerBaseline(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super(TransformerBaseline, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional Encoding for temporal sequences
        self.pos_encoder = nn.Parameter(torch.zeros(1, 120, d_model)) # Support up to 120 months sequence
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Input shape x: (batch_size, seq_len, input_dim)
        seq_len = x.size(1)
        x_proj = self.input_projection(x) # (batch_size, seq_len, d_model)
        x_pos = x_proj + self.pos_encoder[:, :seq_len, :]
        
        tf_out = self.transformer_encoder(x_pos) # (batch_size, seq_len, d_model)
        out = self.fc(tf_out) # (batch_size, seq_len, 1)
        return self.sigmoid(out)

# ----------------------------------------------------
# Training helper functions
# ----------------------------------------------------
def train_pytorch_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training on device: {device}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                
                preds = (outputs > 0.5).cpu().numpy().flatten()
                targets = batch_y.cpu().numpy().flatten()
                all_preds.extend(preds)
                all_targets.extend(targets)
                
        val_loss /= len(val_loader.dataset)
        val_f1 = f1_score(all_targets, all_preds, zero_division=0)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.3f}")

if __name__ == "__main__":
    print("Testing baselines script...")
    
    # 1. Test Random Forest
    X_train = np.random.randn(100, 15)
    y_train = np.random.randint(0, 2, 100)
    rf_base = RandomForestBaseline()
    rf_base.fit(X_train, y_train)
    rf_base.evaluate(X_train, y_train)
    
    # 2. Test LSTM
    dummy_seq = torch.randn(8, 24, 15)
    lstm = LSTMBaseline(input_dim=15, hidden_dim=32)
    out_lstm = lstm(dummy_seq)
    print("LSTM Output Shape:", out_lstm.shape)
    
    # 3. Test Transformer
    transformer = TransformerBaseline(input_dim=15)
    out_tf = transformer(dummy_seq)
    print("Transformer Output Shape:", out_tf.shape)
    print("All baselines compiled and executed successfully under uv!")
