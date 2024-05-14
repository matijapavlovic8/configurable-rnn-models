import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, embeddings, hidden_size=300, num_layers=2):
        super(LSTMModel, self).__init__()
        self.embeddings = embeddings
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm1 = nn.LSTM(embeddings.embedding_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, 150, num_layers=num_layers, batch_first=True)

        self.fc1 = nn.Linear(150, 150)
        self.fc2 = nn.Linear(150, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embeddings(x)
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out.squeeze(1)

