import json
import transformer
import os
import torch
from torch.utils.data import Dataset, DataLoader
from config import max_enc_steps as max_source_length
from config import max_dec_steps as max_target_length
from config import train_data_path,dev_data_path,vocab_path,vocab_size

def read_data_from_json(filename):
    with open("filename", 'r', encoding='utf-8') as load_f:
        temp = load_f.readlines()
        for line in temp:
            dic = json.loads(line)
            article.append(dic['content'])
            summary.append(dic['summary'])
    return article,summary


class Tokenizer:
    def __init__(self):
        self.word2id = {}
        self.id2word = {}
        self.id2word = {}
        self.cur_word = 0
        self.add_word(PAD)
        self.add_word(SENTENCE_START)
        self.add_word(SENTENCE_END)
        self.add_word(OOV)
        
    def add_word(self,word):
        if word not in self.word2id:
            self.id2word[self.cur_word] = word
            self.word2id[word] = self.cur_word
            self.cur_word += 1
    
    def add_sentence(self,sentence):
        for i in sentence:
            if self.cur_word < vocab_size:
                self.add_word(i)
            else:
                return

    def export(self):
        f = open(vocab_path, 'w',encoding='utf-8')
        for i in self.word2id:
            f.write(i)
            f.write('\n')
        f.close()

    def load(self):
        f = open(vocab_path, 'r',encoding='utf-8')
        lines = f.readlines()
        for i in lines:
            word = i.strip()
            self.add_word(word)

        

