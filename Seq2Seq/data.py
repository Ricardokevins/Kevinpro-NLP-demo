import json
import transformer
import taskConfig
import torch
from torch.utils.data import Dataset, DataLoader

def build_tokenizer():
    # corpus = []
    # with open("train.json", 'r', encoding='utf-8') as load_f:
    #     temp = load_f.readlines()
    #     for line in temp:
    #         dic = json.loads(line)
    #         corpus.append(dic['source'])
    model = transformer.Transformer(n_src_vocab= 50000,max_length=500)
    tokenizer = model.get_tokenzier([""])
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
    true_length = len(tokens)
    while len(ids) < length:
        ids.append(0)
    mask = []
    for i in ids:
        if i == 0:
            mask.append(0)
        else:
            mask.append(1)
    return ids,mask,true_length

class SumDataset(Dataset):
    def __init__(self):
        self.source_sents = []
        self.target_sents = []

        self.source_lengths = []
        self.target_lengths = []

        self.source_masks = []
        self.target_masks = []
        with open("train.json", 'r', encoding='utf-8') as load_f:
            temp = load_f.readlines()
            temp=temp[:10000]
            print("dataNumber",len(temp))
            for line in temp:
                dic = json.loads(line)
                source_id,source_mask,source_length = origin_sent2id(dic['source'],taskConfig.padding_source_length)
                target_id, target_mask, target_length = origin_sent2id(dic['target'], taskConfig.padding_target_length)
                self.source_sents.append(source_id)
                self.source_lengths.append(source_length)
                self.source_masks.append(source_mask)

                self.target_sents.append(target_id)
                self.target_lengths.append(target_length)
                self.target_masks.append(target_mask)

                
    def __len__(self):
        return len(self.source_sents)
 
    def __getitem__(self, idx):
        return torch.tensor(self.source_sents[idx]), torch.tensor(self.target_sents[idx]), torch.tensor(self.source_lengths[idx]), torch.tensor(self.target_lengths[idx]), torch.tensor(self.source_masks[idx]), torch.tensor(self.target_masks[idx])
        

import jieba

def get_vocab():
    return tokenzier



data = SumDataset()
# print(data.__getitem__(0))
loader = DataLoader(data, batch_size=1, shuffle=False)
for a,b,c,d,e,f in loader:
    print(a.shape,b.shape)
    print(c.shape,d.shape)
    print(e.shape,f.shape)
    exit()