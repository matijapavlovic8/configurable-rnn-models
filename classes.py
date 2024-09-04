import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from collections import Counter
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.nn import Embedding

@dataclass
class Instance:
    text: list
    label: str


class NLPDataset(Dataset):
    def __init__(self, text_vocab, label_vocab, file_path):
        super(NLPDataset, self).__init__()
        self.text_vocab = text_vocab    
        self.label_vocab = label_vocab
        instances = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                text, label = line.strip().rsplit(', ', 1)
                tokens = text.split()
                instances.append(Instance(tokens, label))
        
        self.instances = instances        

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance = self.instances[idx]
        text_num = self.text_vocab.encode(instance.text)
        label_num = self.label_vocab.encode([instance.label])
        return text_num, label_num


class Vocab:
    def __init__(self, frequencies, max_size=-1, min_freq=0, vocab_type="text"):
        self.itos = []
        self.stoi = {}
        if vocab_type == "text":
            self.itos = ['<PAD>', '<UNK>']
            self.stoi = {'<PAD>': 0, '<UNK>': 1}

        sorted_tokens = sorted(frequencies.items(), key=lambda x: -x[1])
        for token, freq in sorted_tokens:
            if freq < min_freq:
                continue
            if max_size != -1 and len(self.itos) >= max_size:
                break
            self.stoi[token] = len(self.itos)
            self.itos.append(token)

    def encode(self, tokens):
        output = []
        for token in tokens:
            if token in self.stoi:
                output.append(self.stoi[token])
            else:
                output.append(self.stoi['<UNK>'])
        
        return torch.tensor(output)
        
        # return torch.tensor([self.stoi.get(token, self.stoi['<UNK>']) for token in tokens])



def load_embeddings(vocab, embedding_dim=300, embedding_file='data/sst_glove_6b_300d.txt'):
    embeddings = torch.randn(len(vocab.itos), embedding_dim)
    embeddings[0] = torch.zeros(embedding_dim)  # <PAD> token
    
    with open(embedding_file, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            vector = torch.tensor([float(val) for val in values[1:]], dtype=torch.float32)
            if word in vocab.stoi:
                embeddings[vocab.stoi[word]] = vector
                
    return Embedding.from_pretrained(embeddings, freeze=True, padding_idx=0)




def pad_collate_fn(batch, pad_index=0):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(text) for text in texts])
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=pad_index)
    labels = torch.tensor(labels)
    return texts_padded, (labels, lengths)



def get_token_frequencies(file_path):
    counter1 = Counter()
    counter2 = Counter()	
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip().split(', ')
            text = line[0]
            label = line[1]
            tokens = text.split()
            counter1.update(tokens)
            counter2.update([label])
    return counter1, counter2





