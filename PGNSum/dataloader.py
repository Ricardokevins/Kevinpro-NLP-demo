import json
import transformer
import os
import torch
from torch.utils.data import Dataset, DataLoader


SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

max_target_length = 100
max_source_length = 200



def data_prep2json():
    f = open('data/train_art_sum_prep.txt', 'r',encoding='utf-8')
    lines = f.readlines()
    index = 0
    train_samples = []
    while index < len(lines):
        single_sample = {}
        lines[index] = lines[index].replace('\n', '')
        lines[index+1] = lines[index+1].replace('\n','')
        single_sample['source'] =lines[index]
        single_sample['target'] = lines[index + 1]
        index += 2
        train_samples.append(single_sample)
    
    with open("train.json", "w",encoding='utf-8') as dump_f:
        for i in train_samples:
            a = json.dumps(i, ensure_ascii=False)
            dump_f.write(a)
            dump_f.write("\n")



def build_tokenizer():
    corpus = []
    with open("data/train.json", 'r', encoding='utf-8') as load_f:
        temp = load_f.readlines()
        for line in temp:
            dic = json.loads(line)
            corpus.append(dic['source'])
    model = transformer.Transformer(n_src_vocab= 50000,max_length=500)
    tokenizer = model.get_tokenzier(corpus)
    return tokenizer

tokenzier = build_tokenizer()

def origin_sent2id(sent,length):
    ids = []
    tokens = sent.split(' ')
    tokens = tokens[:length-1]
    for i in tokens:
        if i in tokenzier.word2idx:
            ids.append(tokenzier.word2idx[i])
        else:
            ids.append(1)
    while len(ids) < length:
        ids.append(0)

    return ids



def source2id(sent):
    ids = []
    oovs = []

    #tokens = tokenzier.cut(sent)
    tokens = sent.split(' ')
    tokens = tokens[:max_source_length - 1]
    for i in tokens:
        if i in tokenzier.word2idx:
            ids.append(tokenzier.word2idx[i])
        else:
            if i not in oovs:
                oovs.append(i)
            oov_index = oovs.index(i)
            ids.append(50000 + oov_index)
    true_len = len(ids)
    while len(ids) < max_source_length:
        ids.append(0)
    
    return ids,oovs,true_len

def target2id(sent,oovs):
    ids = []
    #tokens = tokenzier.cut(sent)
    tokens = sent.split(' ')
    tokens = tokens[:max_target_length - 1]
    for i in tokens:
        if i in tokenzier.word2idx:
            ids.append(tokenzier.word2idx[i])
        else:
            if i in oovs:
                ids.append(50000 + oovs.index(i))
            else:
                ids.append(1)
    true_len = len(ids)
    while len(ids) < max_target_length:
        ids.append(0)
    return ids,true_len

class SumDataset(Dataset):
    def __init__(self):
        self.source_sent = []
        self.target_sent = []
        self.source_sent_ext = []
        self.target_sent_ext = []
        self.max_oov_num = []

        self.source_length = []
        self.target_length = []
        with open("data/train.json", 'r', encoding='utf-8') as load_f:
            temp = load_f.readlines()
            temp=temp[:10000]
            print("dataNumber",len(temp))
            for line in temp:
                dic = json.loads(line)
                ids, oovs, source_len = source2id(dic['source'])
                self.source_length.append(source_len)
                self.source_sent_ext.append(ids)
                ids, target_len = target2id(dic['target'], oovs)
                self.target_length.append(target_len)
                self.target_sent_ext.append(ids)
                self.max_oov_num.append(len(oovs))
                self.source_sent.append(origin_sent2id(dic['source'],length=max_source_length))
                self.target_sent.append(origin_sent2id(dic['target'],length=max_target_length))
    def __len__(self):
        return len(self.source_sent)
 
    def __getitem__(self, idx):
        return torch.tensor(self.source_sent[idx]),torch.tensor(self.target_sent[idx]),torch.tensor(self.source_sent[idx]),torch.tensor(self.target_sent[idx]),torch.tensor(self.max_oov_num[idx]),torch.tensor(self.source_length[idx]),torch.tensor(self.target_length[idx])

import jieba

def get_vocab():
    return tokenzier

def load_test_set():
    test_set = '前不久 家住 青海 的 高 爸爸 因 事故 入院'
    # words = jieba.cut(test_set)
    # test_set = " ".join(words)
    ids, oovs, source_len=source2id(test_set)
    source_sent_ext=ids     
    max_oov_num=len(oovs)
    source_sent=origin_sent2id(test_set,length=200)
    return torch.tensor([source_sent for i in range(5)]),torch.tensor([source_sent_ext for i in range(5)]),torch.tensor([max_oov_num for i in range(5)]),torch.tensor([source_len for i in range(5)]),test_set,oovs,tokenzier
    #return torch.tensor([source_sent]),torch.tensor([source_sent_ext]),torch.tensor([max_oov_num]),torch.tensor([source_len]),test_set,oovs,tokenzier
# data = MyDataset()
# # print(data.__getitem__(0))
# loader = DataLoader(data, batch_size=1, shuffle=False)
# for x, y ,z in loader:
#     print(x)
#     print(x.shape)
#     print(y)
#     print(y.shape)
#     exit()