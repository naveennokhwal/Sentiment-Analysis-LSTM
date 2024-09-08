import torch
import torch.nn as nn
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    def __init__(self, vectors, labels):
        self.vectors = torch.FloatTensor(vectors)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.vectors[idx], self.labels[idx]

class LSTMSentiment(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.5):
        super(LSTMSentiment, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, (h_n, _) = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        out = self.fc(h_n[-1])  # Use the last layer's hidden state
        out = self.sigmoid(out)
        return out
