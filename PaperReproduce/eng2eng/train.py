import random
import os

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import numpy 
from EasyTransformer.util import ProgressBar
from preprocess import Tokenizer
from preprocess import read_file
from preprocess import SENTENCE_START
from preprocess import SENTENCE_END
from preprocess import PAD
from preprocess import OOV
from preprocess import MAX_Sentence_length
from model import EncoderRNN
from model import DecoderRNN
tokenizer = Tokenizer()

EPOCH = 5
BATCH_SIZE = 64
HIDDEN_SIZE = 256
ENCODER_LAYER = 1
DROP_OUT = 0.1
clip = 2.0
learning_rate = 0.01
decoder_learning_ratio = 5.0
SOS = tokenizer.word2id[SENTENCE_START]
EOS = tokenizer.word2id[SENTENCE_END]


class Seq2SeqDataset(Dataset):
    def __init__(self):
        sent1s, sent2s = read_file("./formatted_movie_lines.txt")
        self.source = []
        self.target = []

        for i, j in zip(sent1s, sent2s):
            source_ids = self.get_source_ids(i)
            self.source.append(source_ids)
            target_ids = self.get_target_ids(j)
            self.target.append(target_ids)

    def get_source_ids(self, article):
        ids = []
        for i in article.split(' '):
            if i in tokenizer.word2id:
                ids.append(tokenizer.word2id[i])
        ids = self.trim_vector(ids)
        ids = [tokenizer.word2id[SENTENCE_START]] + ids
        ids = self.padding_vector(ids)
        return ids

    def get_target_ids(self, article):
        ids = []
        for i in article.split(' '):
            if i in tokenizer.word2id:
                ids.append(tokenizer.word2id[i])
        ids = self.trim_vector(ids)
        ids = ids + [tokenizer.word2id[SENTENCE_END]]
        ids = self.padding_vector(ids)
        return ids

    def trim_vector(self, vector):
        vector = vector[:MAX_Sentence_length]
        return vector

    def padding_vector(self, vector):
        while len(vector) < MAX_Sentence_length+1:
            vector.append(tokenizer.word2id[PAD])
        return vector

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        return torch.tensor(self.source[idx]), torch.tensor(self.target[idx])

def convert(data):
    data.sort(key=lambda x: len(x), reverse=True)
    data = pad_sequence(data, batch_first=True, padding_value=0)
    return data

def collate_fn(data):
    sources = []
    targets = []
    for i in data:
        source = i[0]
        target = i[1]

    source = list(data[0])
    target = list(data[1])
    source = convert(source)
    target = convert(target)
    return source,target

def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.cuda()
    return loss, nTotal.item()

def train():
    dataset = Seq2SeqDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    vocab_size = len(tokenizer.word2id)
    embedding = nn.Embedding(vocab_size, HIDDEN_SIZE)
    encoder = EncoderRNN(HIDDEN_SIZE, embedding, ENCODER_LAYER, DROP_OUT).cuda()
    decoder = DecoderRNN(embedding, HIDDEN_SIZE, vocab_size, ENCODER_LAYER, DROP_OUT).cuda()
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    pbar = ProgressBar(n_total=len(dataloader), desc='Training')

    
    
    for epoch in range(EPOCH):
        print("START EPOCH {}".format(epoch))
        count = 0
        avg_loss = 0
        for source, target in dataloader:
            source = source.cuda()
            target = target.cuda()
            #print(source)
            #print(target)
            # print(source.shape)
            # exit()
            # for i in source:
            #     for j in i:
            #         print(tokenizer.id2word[j.cpu().numpy().tolist()])

            outputs, hidden = encoder(source)
            decoder_input = torch.LongTensor([[SOS] for _ in range(BATCH_SIZE)])
            decoder_input = decoder_input.cuda()
            decoder_hidden = hidden
            decoder_mask = numpy.zeros((BATCH_SIZE, MAX_Sentence_length+1))
             
            for i in range(len(target)):
                for j in range(len(target[i])):
                    if target[i][j] != tokenizer.word2id[PAD]:
                        decoder_mask[i][j] = 1

            decoder_mask = torch.BoolTensor(decoder_mask).cuda()
            step_losses = []
            #use_teacher_forcing = True if random.random() < 0.5 else False
            use_teacher_forcing = True
            if use_teacher_forcing:
                for i in range(MAX_Sentence_length+1):
                    #print(decoder_input)
                    decoder_outputs, hidden = decoder(decoder_input, decoder_hidden, outputs)
                    decoder_input = target[:,i].view(-1, 1)
                    # print(decoder_input.shape)
                    # exit()
                    decoder_input = decoder_input.cuda()
                    step_mask = decoder_mask[:, i]
                    gold_probs = torch.gather(decoder_outputs, 1, target[:, i].view(-1, 1)).squeeze(1)
                    step_loss = -torch.log(gold_probs + 1e-12)
                    step_loss = step_loss * step_mask
                    step_losses.append(step_loss)
                #exit()
            else:
                for i in range(MAX_Sentence_length+1):
                    decoder_outputs, hidden = decoder(decoder_input, decoder_hidden, outputs)
                    _, topi = decoder_outputs.topk(1)
                    decoder_input = torch.LongTensor([[topi[i][0]] for i in range(BATCH_SIZE)])
                    
                    decoder_input = decoder_input.cuda()
                    step_mask = decoder_mask[:, i]
                    gold_probs = torch.gather(decoder_outputs, 1, target[:, i].view(-1, 1)).squeeze(1)
                    #print(gold_probs)
                    step_loss = -torch.log(gold_probs + 1e-12)
                    step_loss = step_loss * step_mask
                    step_losses.append(step_loss)
            #print(step_losses)
            sum_losses = torch.sum(torch.stack(step_losses, 0), 0)
            avg_len = torch.sum(decoder_mask) / BATCH_SIZE
            sum_losses = sum_losses / avg_len
            #print(sum_losses)
            loss = torch.mean(sum_losses)
            loss.backward()
            _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
            _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
            #exit()
            # Adjust model weights
            encoder_optimizer.step()
            decoder_optimizer.step()

            avg_loss += loss
            count += 1
            pbar(count, {'loss': loss})

        torch.save({
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'embedding': embedding.state_dict()
            }, 'Epoch_{}_{}.tar'.format(epoch, 'checkpoint'))


def test():
    dataset = Seq2SeqDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    vocab_size = len(tokenizer.word2id)
    pbar = ProgressBar(n_total=len(dataloader), desc='Training')

    LOAD_FILE_NAME = './Epoch_0_checkpoint.tar'
    checkpoint = torch.load(LOAD_FILE_NAME)
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    embedding_sd = checkpoint['embedding']

    embedding = nn.Embedding(vocab_size, HIDDEN_SIZE)
    embedding.load_state_dict(embedding_sd)

    encoder = EncoderRNN(HIDDEN_SIZE, embedding, ENCODER_LAYER, DROP_OUT).cuda()
    encoder.load_state_dict(encoder_sd)
    
    decoder = DecoderRNN(embedding, HIDDEN_SIZE, vocab_size, ENCODER_LAYER, DROP_OUT).cuda()
    decoder.load_state_dict(decoder_sd)

    encoder = encoder.cuda()
    decoder = decoder.cuda()
    encoder.eval()
    decoder.eval()

    avg_loss = 0
    count = 0
   
    for source, target in dataloader:
        source = source.cuda()
        target = target.cuda()
        outputs, hidden = encoder(source)
        decoder_input = torch.LongTensor([[SOS] for _ in range(BATCH_SIZE)])
        decoder_input = decoder_input.cuda()
        decoder_hidden = hidden
        decoder_mask = torch.zeros((BATCH_SIZE, MAX_Sentence_length+1)).cuda()
    
        for i in range(len(target)):
            for j in range(len(target[i])):
                if target[i][j] != tokenizer.word2id[PAD]:
                    decoder_mask[i][j] = 1
        step_losses = []
        decode_result_ids = []
        for i in range(MAX_Sentence_length):
            decoder_outputs, hidden = decoder(decoder_input, decoder_hidden, outputs)
            prob, topi = decoder_outputs.topk(1)
            decode_result_ids.append(topi[i][0].cpu().numpy().tolist())
            decoder_input = torch.LongTensor([[topi[i][0]] for i in range(BATCH_SIZE)])
            decoder_input = decoder_input.cuda()
            step_mask = decoder_mask[:, i]
        
        # print(decode_result_ids)
        # exit()
        pbar(count, {})



def main():
    train()
    #test()


main()