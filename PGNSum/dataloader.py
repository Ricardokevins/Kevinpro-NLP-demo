import json
import transformer
import os
import torch
from torch.utils.data import Dataset, DataLoader


SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

max_length = 128




def data_prep2json():
    f = open('data/train_art_sum_prep.txt', 'r',encoding='utf-8')
    lines = f.readlines()
    index = 0
    train_samples = []
    while index < len(lines):
        single_sample = {}
        lines[index] = lines[index].replace('\n', '')
        lines[index+1] = lines[index].replace('\n','')
        single_sample['source'] =SENTENCE_START+' '+ lines[index]+' '+SENTENCE_END
        single_sample['target'] = SENTENCE_START+' '+lines[index + 1]+' '+SENTENCE_END
        index += 2
        train_samples.append(single_sample)
    
    with open("train.json", "w",encoding='utf-8') as dump_f:
        for i in train_samples:
            a = json.dumps(i, ensure_ascii=False)
            dump_f.write(a)
            dump_f.write("\n")

#data_prep2json()

def build_tokenizer():
    corpus = []
    with open("train.json", 'r', encoding='utf-8') as load_f:
        temp = load_f.readlines()
        for line in temp:
            dic = json.loads(line)
            corpus.append(dic['source'])
    model = transformer.Transformer(n_src_vocab= 50000,max_length=max_length)
    tokenizer = model.get_tokenzier(corpus)
    return tokenizer

tokenzier = build_tokenizer()


def source2id(sent):
    ids = []
    oovs = []

    #tokens = tokenzier.cut(sent)
    tokens = sent.split(' ')
    for i in tokens:
        if i in tokenzier.word2idx:
            ids.append(tokenzier.word2idx[i])
        else:
            if i not in oovs:
                oovs.append(i)
            oov_index = oovs.index(i)
            ids.append(50000 + oov_index)
    while len(ids) < max_length:
        ids.append(0)
    return ids,oovs

def target2id(sent,oovs):
    ids = []
    #tokens = tokenzier.cut(sent)
    tokens = sent.split(' ')
    for i in tokens:
        if i in tokenzier.word2idx:
            ids.append(tokenzier.word2idx[i])
        else:
            if i in oovs:
                ids.append(50000 + oovs.index(i))
            else:
                ids.append(1)
    while len(ids) < 128:
        ids.append(0)
    return ids

class SumDataset(Dataset):
    def __init__(self):
        self.source_sent = []
        self.target_sent = []
        self.max_oov_num = []
        with open("train.json", 'r', encoding='utf-8') as load_f:
            temp = load_f.readlines()
            temp=temp[:2000]
            for line in temp:
                dic = json.loads(line)
                ids,oovs=source2id(dic['source'])
                self.source_sent.append(ids)
                self.target_sent.append(target2id(dic['target'],oovs))
                self.max_oov_num.append(len(oovs))
 
    def __len__(self):
        return len(self.source_sent)
 
    def __getitem__(self, idx):
        return torch.tensor(self.source_sent[idx]),torch.tensor(self.target_sent[idx]),torch.tensor(self.max_oov_num[idx])



# data = MyDataset()
# # print(data.__getitem__(0))
# loader = DataLoader(data, batch_size=2, shuffle=False)
# for x, y ,z in loader:
#     print(x)
#     print(x.shape)
#     print(y)
#     print(y.shape)
#     exit()