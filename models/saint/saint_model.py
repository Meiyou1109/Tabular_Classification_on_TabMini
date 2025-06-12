import torch
import torch.nn as nn

class SAINTModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_heads=2, num_layers=1, dropout=0.1):
        super(SAINTModel, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim,
                                                   nhead=num_heads,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Binary classification
        )

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # [B, 1, H]
        x = self.transformer(x)             # [B, 1, H]
        x = x.squeeze(1)                    # [B, H]
        out = self.mlp(x)                   # [B, 2]
        return out
