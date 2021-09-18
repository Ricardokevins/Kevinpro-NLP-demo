import random
import os

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from preprocess import Tokenizer
from preprocess import read_file
from preprocess import SENTENCE_START
from preprocess import SENTENCE_END
from preprocess import PAD
from preprocess import OOV
from preprocess import MAX_Sentence_length
from model import EncoderRNN

tokenizer = Tokenizer()

BATCH_SIZE = 32
HIDDEN_SIZE = 64
ENCODER_LAYER = 1
DROP_OUT = 0.1

class Seq2SeqDataset(Dataset):
    def __init__(self):
        sent1s, sent2s = read_file("./formatted_movie_lines.txt")
        self.source = []
        self.target = []
        for i, j in zip(sent1s, sent2s):
            source_ids = self.get_ids(i)
            source_ids = [tokenizer.word2id[SENTENCE_START]] + source_ids
            self.source.append(source_ids)
            target_ids = self.get_ids(j)
            target_ids = target_ids + [tokenizer.word2id[SENTENCE_END]]
            self.target.append(target_ids)

    def get_ids(self,article):
        ids = []
        for i in article.split(' '):
            if i in tokenizer.word2id:
                ids.append(tokenizer.word2id[i])
        return self.trim_vector(ids)

    def trim_vector(self,vector):
        while len(vector) < MAX_Sentence_length:
            vector.append(tokenizer.word2id[PAD])
        vector = vector[:MAX_Sentence_length]
        return vector

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        return torch.tensor(self.source[idx]), torch.tensor(self.target[idx])


def train():
    dataset = Seq2SeqDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    vocab_size = len(tokenizer.word2id)
    embedding = nn.Embedding(vocab_size, HIDDEN_SIZE)
    encoder = EncoderRNN(HIDDEN_SIZE, embedding, ENCODER_LAYER, DROP_OUT)
    for source,target in dataloader:
        encode_result = encoder(source)

train()