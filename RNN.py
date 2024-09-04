import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from functions import train, evaluate
from classes import NLPDataset, Vocab, pad_collate_fn, load_embeddings, get_token_frequencies
from torch.utils.data import DataLoader
import json


class RNNModel(nn.Module):
    def __init__(self, embeddings, hidden_size=150, num_layers=2, rnn_type='gru', dropout=0.2, bidirectional=False):
        super(RNNModel, self).__init__()
        self.embeddings = embeddings
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn_layers = nn.ModuleList()

        rnn_dict = {
            'lstm': nn.LSTM,
            'gru': nn.GRU,
            'vanilla': nn.RNN
        }
        rnn_class = rnn_dict.get(rnn_type)

        for i in range(num_layers):
            input_size = embeddings.weight.shape[1] if i == 0 else hidden_size * (2 if bidirectional else 1)
            if rnn_type == "vanilla":
                layer = rnn_class(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers != 1 else 0,
                    bidirectional=bidirectional,
                    batch_first=True,
                    nonlinearity='tanh'
                )
            elif rnn_type is not None:
                layer = rnn_class(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers != 1 else 0,
                    bidirectional=bidirectional,
                    batch_first=True
                )
            else:
                raise ValueError(f"Unsupported rnn_type: {rnn_type}")
            self.rnn_layers.append(layer)

        self.fc1 = nn.Linear(hidden_size * (2 if bidirectional else 1), hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embeddings(x)
        for layer in self.rnn_layers:
            x, _ = layer(x)

        x = x[:, -1, :]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def main():
    seed = 7052020
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    frequencies1, frequencies2 = get_token_frequencies("data/sst_train_raw.csv")
    label_vocab = Vocab(frequencies2, min_freq=1, vocab_type='label')
    text_vocab = Vocab(frequencies1, min_freq=1)

    train_dataset = NLPDataset(text_vocab, label_vocab, 'data/sst_train_raw.csv')
    valid_dataset = NLPDataset(text_vocab, label_vocab, 'data/sst_valid_raw.csv')
    test_dataset = NLPDataset(text_vocab, label_vocab, 'data/sst_test_raw.csv')

    batch_size_train = 10
    batch_size_eval = 32
    num_epochs = 5

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, collate_fn=pad_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size_eval, shuffle=False, collate_fn=pad_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_eval, shuffle=False, collate_fn=pad_collate_fn)

    embeddings = load_embeddings(text_vocab)

    criterion = nn.BCEWithLogitsLoss()

    with open("config2.json", "r") as file:
        config = json.load(file)

    for idx, config in enumerate(config["hyperparameters"]):
        if idx != 2:
            continue
        print(f"Configuration {idx + 1}:")
        string = ""
        file_name = f"Configuration {idx + 1}"

        model = RNNModel(embeddings, hidden_size=config["hidden_size"], bidirectional=config["bidirectional"],
                         rnn_type=config["rnn_type"], num_layers=config["num_layers"], dropout=config["dropout"]).to(
            device)

        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
        for epoch in range(num_epochs):
            train_loss = train(model, train_loader, optimizer, criterion, device, grad_clip=config["gradient_clip"])
            valid_loss, valid_acc, valid_f1, valid_conf_matrix = evaluate(model, valid_loader, criterion, device)

            string += f'Epoch {epoch + 1}:' + "\n"
            string += f'  Train Loss: {train_loss:.4f}' + "\n"
            string += f'  Valid Loss: {valid_loss:.4f}' + "\n"
            string += f'  Valid Accuracy: {valid_acc:.4f}' + "\n"
            string += f'  Valid F1 Score: {valid_f1:.4f}' + "\n"
            string += f'  Valid Confusion Matrix:\n{valid_conf_matrix}' + "\n"
            string += "\n"

        test_loss, test_acc, test_f1, test_conf_matrix = evaluate(model, test_loader, criterion, device)
        string += 'Test Set Performance:' + "\n"
        string += f'  Test Loss: {test_loss:.4f}' + "\n"
        string += f'  Test Accuracy: {test_acc:.4f}' + "\n"
        string += f'  Test F1 Score: {test_f1:.4f}' + "\n"
        string += f'  Test Confusion Matrix:\n{test_conf_matrix}' + "\n"
        string += "\n"

        with open(f"{file_name}.txt", "w") as file:
            file.write(string)


if __name__ == '__main__':
    main()
