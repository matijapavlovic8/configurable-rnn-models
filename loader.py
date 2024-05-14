import pandas as pd
import numpy as np
import torch
from torch.nn import Embedding
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset

PAD = '<PAD>'
UNK = '<UNK>'


class Instance:
    def __init__(self, input, label):
        self.input = input
        self.label = label


class Vocab:
    def __init__(self, frequencies, max_size=-1, min_freq=0, target=False):
        self.itos = {}
        self.stoi = {}
        if not target:
            self.itos = {0: PAD, 1: UNK}
            self.stoi = {PAD: 0, UNK: 1}
        self.max_size = max_size
        self.min_freq = min_freq

        sorted_frequencies = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
        idx = len(self.stoi)
        for token, freq in sorted_frequencies:
            if max_size != -1 and idx > max_size:
                break
            if freq >= min_freq:
                self.itos[idx] = token
                self.stoi[token] = idx
                idx += 1
            else:
                break

    def __len__(self):
        return len(self.stoi)

    def encode(self, input):
        output = []

        for i in input:
            if self.stoi.keys().__contains__(i):
                output.append(self.stoi[i])
            else:
                output.append(self.stoi[UNK])
        return torch.tensor(output)


class NLPDataset(Dataset):
    def __init__(self, input_vocabulary, target_vocabulary, instances):
        self.input_vocabulary = input_vocabulary
        self.target_vocabulary = target_vocabulary
        self.instances = instances

    def __getitem__(self, item):
        input = self.instances[item].input
        label = self.instances[item].label
        numericalized_input = self.input_vocabulary.encode(input)
        numericalized_label = self.target_vocabulary.encode([label])
        return numericalized_input, numericalized_label

    def __len__(self):
        return len(self.instances)


def generate_embedding_matrix(input_vocabulary, file=None, d=300):
    v = len(input_vocabulary)
    matrix = torch.randn(v, d)
    matrix[0] = torch.zeros(d)
    if file is not None:
        df = pd.read_csv(file, delimiter=' ', header=None)
        for idx in range(len(df)):
            item = df.iloc[idx, 0]
            if item in input_vocabulary.stoi:
                i = input_vocabulary.stoi[item]
                data = df.iloc[idx, 1:].values
                data = data.astype(np.float32)
                matrix[i] = torch.tensor(data)

    return Embedding.from_pretrained(matrix, padding_idx=0)


def pad_collate_fn(batch, pad_index=0):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(text) for text in texts])
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=pad_index)
    return padded_texts, (torch.tensor(labels), lengths)


def generate_instances(file_path):
    data = pd.read_csv(file_path, delimiter=',', header=None)

    instances = []
    for index, row in data.iterrows():
        input_text, target = row
        instance = Instance(input_text.split(), target.strip())
        instances.append(instance)

    return instances


def generate_frequencies(instances):
    input_word_freq = {}
    target_word_freq = {}
    for instance in instances:
        for word in instance.input:
            input_word_freq[word] = input_word_freq.get(word, 0) + 1
        target_word_freq[instance.label] = target_word_freq.get(instance.label, 0) + 1

    return input_word_freq, target_word_freq
