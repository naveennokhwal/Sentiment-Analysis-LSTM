import torch
import torch.nn as nn
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    def __init__(self, vectors, labels):
        self.vectors = torch.FloatTensor(vectors)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.vectors[idx], self.labels[idx]


class LSTMSentiment(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(LSTMSentiment, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take the last time step output
        out = self.fc(lstm_out)
        return out  # Don't apply softmax here as it's included in CrossEntropyLoss