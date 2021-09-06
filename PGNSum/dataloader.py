import json
import transformer
import os
import torch
from torch.utils.data import Dataset, DataLoader
from config import max_enc_steps as max_source_length
from config import max_dec_steps as max_target_length
from config import vocab_size
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'


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
    model = transformer.Transformer(n_src_vocab= vocab_size , max_length = max_source_length)
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
            ids.append(vocab_size + oov_index)
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
                ids.append(vocab_size + oovs.index(i))
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
            temp=temp[:100000]
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
                self.target_sent.append(origin_sent2id(dic['target'], length=max_target_length))
                
            print(temp[0])
            print(self.source_sent[0])
            print(self.target_sent[0])
            print(self.source_sent_ext[0])
            print(self.target_sent_ext[0])
            print(self.max_oov_num[0])
            print(self.source_length[0])
            print(self.target_length[0])
            

    def __len__(self):
        return len(self.source_sent)
 
    def __getitem__(self, idx):
        return torch.tensor(self.source_sent[idx]),torch.tensor(self.target_sent[idx]),torch.tensor(self.source_sent_ext[idx]),torch.tensor(self.target_sent_ext[idx]),torch.tensor(self.max_oov_num[idx]),torch.tensor(self.source_length[idx]),torch.tensor(self.target_length[idx])

import jieba

def get_vocab():
    return tokenzier

def load_test_set():
    #test_set = '25 日 ， 南宁 某 酒店 一 招牌 掉落 砸碎 玻璃 ， 电视台 一女 记者 前往 报道 ， 用 手机 拍摄 素材 。 这时 冲出 两个 不明 身份 的 男子 ， 阻挠 和 辱骂 记者 ， 扇 了 记者 一巴掌 ， 还 将 其 手机 摔 在 地上 。 记者 报警 后 男子 趁乱 逃跑 ， 酒店 方 否认 与 男子 有关 。 新 京报 我们 视频 的 秒 拍 视频'
     
    test_set = "近日 ， 因 天气 太热 ， 安徽 一 老太 在 买 肉 路上 突然 眼前 一黑 ， 摔倒 在 地 。 她 怕 别人 不 扶 她 ， 连忙 说 \" 快 扶 我 起来 ， 我 不 讹 你 ， 地上 太热 我 要 熟 了 ！ \" 这一喊 周围 人 都 笑 了 ， 老人 随后 被 扶 到 路边 休息 。最近 老人 尽量避免 出门 !"
    # words = jieba.cut(test_set)
    # test_set = " ".join(words)
    ids, oovs, source_len=source2id(test_set)
    source_sent_ext=ids     
    max_oov_num=len(oovs)
    source_sent=origin_sent2id(test_set,length=200)
    #return torch.tensor([source_sent for i in range(5)]),torch.tensor([source_sent_ext for i in range(5)]),torch.tensor([max_oov_num for i in range(5)]),torch.tensor([source_len for i in range(5)]),test_set,oovs,tokenzier
    return torch.tensor([source_sent]),torch.tensor([source_sent_ext]),torch.tensor([max_oov_num]),torch.tensor([source_len]),test_set,oovs,tokenzier
# data = MyDataset()
# # print(data.__getitem__(0))
# loader = DataLoader(data, batch_size=1, shuffle=False)
# for x, y ,z in loader:
#     print(x)
#     print(x.shape)
#     print(y)
#     print(y.shape)
#     exit()