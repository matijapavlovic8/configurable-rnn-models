import torch
from torch import nn


class ConfigurableRNN(nn.Module):
    def __init__(self, embeddings, hidden_size=150, num_layers=2, rnn_type='gru', dropout=0.0,
                 bidirectional=False):
        super(ConfigurableRNN, self).__init__()
        self.embeddings = embeddings
        self.rnn_type = rnn_type.lower()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.rnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            if self.rnn_type == 'lstm':
                rnn_layer = nn.LSTM(input_size=300 if _ == 0 else hidden_size,
                                    hidden_size=hidden_size,
                                    num_layers=1,
                                    dropout=dropout if num_layers != 1 else 0,
                                    bidirectional=bidirectional,
                                    batch_first=True)
            elif self.rnn_type == 'gru':
                rnn_layer = nn.GRU(input_size=300 if _ == 0 else hidden_size,
                                   hidden_size=hidden_size,
                                   num_layers=1,
                                   dropout=dropout if num_layers != 1 else 0,
                                   bidirectional=bidirectional,
                                   batch_first=True)
            elif self.rnn_type == 'vanilla':
                rnn_layer = nn.RNN(input_size=300 if _ == 0 else hidden_size,
                                   hidden_size=hidden_size,
                                   num_layers=1,
                                   nonlinearity='tanh',
                                   dropout=dropout if num_layers != 1 else 0,
                                   bidirectional=bidirectional,
                                   batch_first=True)
            else:
                raise ValueError("Invalid RNN type. Please choose from 'lstm', 'gru', or 'vanilla'.")
            self.rnn_layers.append(rnn_layer)

        self.fc1 = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        embedded = self.embeddings(x)
        for layer_idx in range(self.num_layers):
            embedded, _ = self.rnn_layers[layer_idx](embedded)

        last_hidden_state = embedded[:, -1, :]
        fc_out = self.fc1(last_hidden_state)
        relu_out = self.relu(fc_out)
        output = self.fc2(relu_out)
        return output
