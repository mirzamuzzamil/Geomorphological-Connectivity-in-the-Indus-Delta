import torch
import torch.nn as nn

class LSTMBaseline(nn.Module):
    """
    LSTM baseline model for sequential modeling of surface water dynamics.
    Learns temporal sequence dependencies across monthly time steps.
    """
    def __init__(self, input_dim, hidden_dim=32, num_layers=2, output_dim=1):
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
        """
        x shape: (batch_size, sequence_length, input_dim)
        Output shape: (batch_size, sequence_length, output_dim)
        """
        lstm_out, _ = self.lstm(x) # (batch_size, sequence_length, hidden_dim)
        out = self.fc(lstm_out)   # (batch_size, sequence_length, output_dim)
        return self.sigmoid(out)
