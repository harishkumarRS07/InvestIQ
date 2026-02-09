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
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
        super(LSTMAttentionModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)
        # lstm_out: [batch_size, seq_len, hidden_dim]
        
        context, attn_weights = self.attention(lstm_out)
        # context: [batch_size, hidden_dim]
        
        prediction = self.fc(context)
        # prediction: [batch_size, output_dim]
        
        return prediction
