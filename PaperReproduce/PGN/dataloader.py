import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from config import max_enc_steps
from config import max_dec_steps
from config import train_data_path, dev_data_path, vocab_path, vocab_size
from config import SENTENCE_START, SENTENCE_END, OOV, PAD


def read_data_from_json(filename):
    article, summary = [], []
    with open(filename, 'r', encoding='utf-8') as load_f:
        temp = load_f.readlines()
        for line in temp:
            dic = json.loads(line)
            article.append(dic['content'])
            summary.append(dic['summary'])
    return article, summary


class Tokenizer:
    def __init__(self, LoadFromFile=True):
        self.word2id = {}
        self.id2word = {}
        self.id2word = {}
        self.cur_word = 0
        self.add_word(PAD)
        self.add_word(SENTENCE_START)
        self.add_word(SENTENCE_END)
        self.add_word(OOV)
        if LoadFromFile:
            self.load()

    def add_word(self, word):
        if word not in self.word2id:
            self.id2word[self.cur_word] = word
            self.word2id[word] = self.cur_word
            self.cur_word += 1

    def add_sentence(self, sentence):
        for i in sentence:
            if self.cur_word < vocab_size:
                self.add_word(i)
            else:
                return

    def export(self):
        f = open(vocab_path, 'w', encoding='utf-8')
        for i in self.word2id:
            f.write(i)
            f.write('\n')
        f.close()

    def load(self):
        f = open(vocab_path, 'r', encoding='utf-8')
        lines = f.readlines()
        for i in lines:
            word = i.strip()
            self.add_word(word)


class SumDataset(Dataset):
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.enc_input_list = []
        self.enc_input_ext_list = []
        self.dec_input_list = []
        self.dec_output_list = []
        self.enc_len_list = []
        self.dec_len_list = []
        self.oov_word_num_list = []

        load_f = open(train_data_path, 'r', encoding='utf-8')
        temp = load_f.readlines()
        temp = temp[:10000]
        print("dataNumber", len(temp))
        for line in temp:
            dic = json.loads(line)
            source = dic['content']
            target = dic['summary']
            enc_input, enc_input_ext, enc_len, dec_input, dec_output, dec_len, oov_word_list = self.get_one_sample(source, target)
            self.enc_input_list.append(enc_input)
            self.enc_input_ext_list.append(enc_input_ext)
            self.dec_input_list.append(dec_input)
            self.dec_output_list.append(dec_output)
            self.enc_len_list.append(enc_len)
            self.dec_len_list.append(dec_len)
            self.oov_word_num_list.append(len(oov_word_list))
            #print(self.enc_input_list)

        # For test to see
        # print(self.enc_input_list[0])
        # print(self.enc_input_ext_list[0])
        # print(self.dec_input_list[0])
        # print(self.dec_output_list[0])
        # print(self.enc_len_list[0])
        # print(self.dec_len_list[0])
        # print(self.oov_word_num_list[0])

    def __len__(self):
        return len(self.enc_input_list)

    def __getitem__(self, idx):
        
        return torch.tensor(self.enc_input_list[idx]), torch.tensor(self.enc_input_ext_list[idx]), torch.tensor(self.dec_input_list[idx]), torch.tensor(self.dec_output_list[idx]), torch.tensor(self.enc_len_list[idx]), torch.tensor(self.dec_len_list[idx]), torch.tensor(self.oov_word_num_list[idx])

    def sentence2ids(self, article):
        ids = []
        for i in article:
            if i in self.tokenizer.word2id:
                ids.append(self.tokenizer.word2id[i])
            else:
                ids.append(self.tokenizer.word2id[OOV])
        ids = ids[:max_enc_steps]
        return ids

    def sentence2idsExt(self,article):
        ids = []
        oov_word_list = []
        for i in article:
            if i in self.tokenizer.word2id:
                ids.append(self.tokenizer.word2id[i])
            else:
                if i not in oov_word_list:
                    oov_word_list.append(i)
                oov_index = oov_word_list.index(i)
                ids.append(vocab_size+oov_index)
        ids = ids[:max_enc_steps]
        return ids, oov_word_list

    def target2ids(self, target, oov_word_list):
        ids = []
        oov_word_list = []
        for i in target:
            if i in self.tokenizer.word2id:
                ids.append(self.tokenizer.word2id[i])
            else:
                if i in oov_word_list:
                    oov_index = oov_word_list.index(i)
                    ids.append(vocab_size+oov_index)
                else:
                    ids.append(self.tokenizer.word2id[OOV])
        ids = ids[:max_dec_steps]
        return ids

    def target_process(self, target):
        target_input = [self.tokenizer.word2id[SENTENCE_START]] + target[:]
        target_output = target[:] + [self.tokenizer.word2id[SENTENCE_END]]
        target_input = target_input[:max_dec_steps]
        target_output = target_output[:max_dec_steps]
        assert len(target_input) == len(target_output)
        return target_input, target_output

    def padding_input(self, input_seq):
        while(len(input_seq) < max_enc_steps):
            input_seq.append(self.tokenizer.word2id[PAD])
        return input_seq

    def padding_target(self, output_seq):
        while(len(output_seq) < max_dec_steps):
            output_seq.append(self.tokenizer.word2id[PAD])
        return output_seq

    def get_one_sample(self, source, target):
        enc_input = self.sentence2ids(source)
        dec_input = self.sentence2ids(target)

        # source input use UNK to imply OOV
        # target use OOV_id + max_word_num to imply OOV
        enc_input_ext, oov_word_list = self.sentence2idsExt(source)
        dec_input_ext = self.target2ids(target, oov_word_list)

        dec_input, _ = self.target_process(dec_input)
        _, dec_output = self.target_process(dec_input_ext)

        enc_len = len(enc_input)
        dec_len = len(dec_input)

        enc_input = self.padding_input(enc_input)
        enc_input_ext = self.padding_input(enc_input_ext)
        
        dec_input = self.padding_target(dec_input)
        dec_output = self.padding_target(dec_output)

        assert max_enc_steps == len(enc_input)
        assert max_enc_steps == len(enc_input_ext)
        assert max_dec_steps == len(dec_input)
        assert max_dec_steps == len(dec_output)
        return enc_input, enc_input_ext, enc_len, dec_input, dec_output, dec_len, oov_word_list
