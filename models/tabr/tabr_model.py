import torch
import torch.nn as nn

class TabR(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, rnn_type='gru', use_attention=True):
        super(TabR, self).__init__()
        self.use_attention = use_attention

        self.embedding = nn.Linear(1, hidden_dim)

        if rnn_type == 'gru':
            self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError("Unsupported RNN type")

        if use_attention:
            self.attention = nn.Linear(hidden_dim, 1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: [batch_size, num_features]
        x = x.unsqueeze(-1)  # [B, F, 1]
        x = self.embedding(x)  # [B, F, hidden]
        rnn_out, _ = self.rnn(x)  # [B, F, hidden]

        if self.use_attention:
            attn_weights = torch.softmax(self.attention(rnn_out), dim=1)  # [B, F, 1]
            context = torch.sum(attn_weights * rnn_out, dim=1)  # [B, hidden]
        else:
            context = rnn_out[:, -1, :]  # last hidden

        return self.classifier(context).squeeze(1)
