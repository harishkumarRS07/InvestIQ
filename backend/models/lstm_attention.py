import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        # lstm_output: [batch_size, seq_len, hidden_dim]
        # weights: [batch_size, seq_len, 1]
        weights = F.softmax(self.attn(lstm_output), dim=1)
        # context: [batch_size, hidden_dim]
        context = torch.sum(weights * lstm_output, dim=1)
        return context, weights

class LSTMAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=1, dropout=0.3):
        super(LSTMAttentionModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = True
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )
        
        # Bidirectional outputs 2 * hidden_dim
        lstm_output_dim = hidden_dim * 2 if self.bidirectional else hidden_dim
        
        self.layer_norm = nn.LayerNorm(lstm_output_dim)
        self.attention = Attention(lstm_output_dim)
        
        self.fc1 = nn.Linear(lstm_output_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)
        # lstm_out: [batch_size, seq_len, num_directions * hidden_dim]
        
        lstm_out = self.layer_norm(lstm_out)
        
        context, attn_weights = self.attention(lstm_out)
        # context: [batch_size, num_directions * hidden_dim]
        
        x = F.relu(self.fc1(context))
        x = self.dropout(x)
        prediction = self.fc2(x)
        
        return prediction

