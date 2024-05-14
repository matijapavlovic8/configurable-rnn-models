import json

import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from loader import Vocab, NLPDataset, generate_frequencies, generate_instances, pad_collate_fn, \
    generate_embedding_matrix
from baseline_model import BaselineModel
from lstm_model import LSTMModel
from rnn_model import ConfigurableRNN
from utils import train, evaluate


def run_model_with_config(config, embeddings, train_loader, valid_loader, test_loader, criterion):
    hidden_size = config['hidden_size']
    num_layers = config['num_layers']
    rnn_type = config['rnn_type']
    dropout = config['dropout']
    bidirectional = config['bidirectional']
    lr = config['lr']
    grad_clip = config['gradient_clip']

    rnn_model = ConfigurableRNN(embeddings, hidden_size, num_layers, rnn_type, dropout, bidirectional)
    optimizer = torch.optim.Adam(rnn_model.parameters(), lr=1e-4)
    train(rnn_model, train_loader, valid_loader, criterion, optimizer, gradient_clip=grad_clip, num_epochs=5)

    evaluation_results_file = f"evaluation_results_{rnn_type}_{hidden_size}_{num_layers}_{dropout}_{lr}_{grad_clip}_{bidirectional}.txt"
    evaluate(rnn_model, test_loader, criterion, evaluation_results_file)


def main():
    np.random.seed(7052020)
    torch.manual_seed(7052020)
    instances = generate_instances('data/sst_train_raw.csv')

    with open('config.json', 'r') as f:
        config_data = json.load(f)

    vocab_params = config_data.get('vocab_params', {})
    input_frequencies, target_frequencies = generate_frequencies(instances)
    input_vocab = Vocab(input_frequencies, **vocab_params)
    target_vocab = Vocab(target_frequencies, target=True, **vocab_params)

    train_dataset = NLPDataset(input_vocab, target_vocab, instances)
    valid_dataset = NLPDataset(input_vocab, target_vocab, generate_instances('data/sst_valid_raw.csv'))
    test_dataset = NLPDataset(input_vocab, target_vocab, generate_instances('data/sst_test_raw.csv'))
    embeddings = generate_embedding_matrix(input_vocab, 'data/sst_glove_6b_300d.txt')

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, collate_fn=pad_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=32, collate_fn=pad_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=pad_collate_fn)
    criterion = BCEWithLogitsLoss()

    # baseline_model = BaselineModel(embeddings)
    # optimizer = torch.optim.Adam(baseline_model.parameters(), lr=1e-4)
    # train(baseline_model, train_loader, valid_loader, criterion, optimizer)
    # evaluate(baseline_model, test_loader, criterion, 'evaluation_result_baseline.txt')
    #
    # lstm_model = LSTMModel(embeddings)
    # optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-4)
    # train(lstm_model, train_loader, valid_loader, criterion, optimizer, num_epochs=5)
    # evaluate(lstm_model, test_loader, criterion, 'evaluation_results_lstm.txt')

    for idx, config in enumerate(config_data['hyperparameters']):
        print(f"Running configuration {idx + 1}")
        run_model_with_config(config, embeddings, train_loader, valid_loader, test_loader, criterion)
        print(f"Configuration {idx + 1} completed.")


if __name__ == '__main__':
    main()
