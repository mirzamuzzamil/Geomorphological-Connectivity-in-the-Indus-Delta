import torch
import torch.nn as nn

class TransformerBaseline(nn.Module):
    """
    Transformer baseline model for spatio-temporal modeling of surface water dynamics.
    Applies self-attention across monthly time steps.
    """
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, max_seq_len=120):
        super(TransformerBaseline, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional Encoding for temporal sequences
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
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
        """
        x shape: (batch_size, sequence_length, input_dim)
        Output shape: (batch_size, sequence_length, 1)
        """
        seq_len = x.size(1)
        x_proj = self.input_projection(x) # (batch_size, sequence_length, d_model)
        x_pos = x_proj + self.pos_encoder[:, :seq_len, :]
        
        tf_out = self.transformer_encoder(x_pos) # (batch_size, sequence_length, d_model)
        out = self.fc(tf_out) # (batch_size, sequence_length, 1)
        return self.sigmoid(out)
