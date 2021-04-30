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
    #print(tokens)
    #print(true_length)
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
        return torch.tensor(self.source_sents[idx]), torch.tensor(self.target_sents[idx]), torch.tensor(self.source_lengths[idx]), torch.tensor(self.target_lengths[idx]), torch.BoolTensor(self.source_masks[idx]), torch.BoolTensor(self.target_masks[idx])
        
def convertid2sentence(tokens):
    sentence = []
    for i in tokens:
        id = i.item()
        if id in tokenzier.idx2word:
            sentence.append(tokenzier.idx2word[id])
        else:
            sentence.append('oov')
    return sentence

def load_test_set():
    #input = '盖 被子 ， 摇 摇篮 ， 汪星 人 简直 要 把 萌娃 宠 上天 ～ 细致 周到 有 耐心 ， 脾气 还好 ， 汪星 人 不愧 是 一届 带娃 好手 [ 笑 而 不语 ] 偶买 噶 视频 的 秒 拍 视频'
    input = '徐州 18 岁 农家 女孩 宋爽 ， 今年 考入 清华大学 。 除了 自己 一路 闯关 ， 年 年 拿 奖 ， 还 帮 妹妹 、 弟弟 制定 学习 计划 ， 姐弟 仨 齐头并进 ， 妹妹 也 考上 区里 最好 的 中学 。 这个 家里 的 收入 ， 全靠 父亲 务农 和 打零工 ， 但 宋爽 懂事 得 让 人 心疼 ， 曾 需要 200 元 奥数 竞赛 的 教材费 ， 她 羞于 开口 ， 愣 是 急 哭 了 ...   戳 腾讯 公益 帮帮 她们 ！ # 助学 圆梦 #   江苏 新闻 的 秒 拍 视频'
    source_id,source_mask,source_length = origin_sent2id(input,taskConfig.padding_source_length)
    #target_id, target_mask, target_length = origin_sent2id(dic['target'], taskConfig.padding_target_length)
    print(source_id)
    print(source_length)
    return torch.tensor([source_id]), torch.tensor(source_mask), torch.tensor([source_length])

import jieba

def get_vocab():
    return tokenzier



# data = SumDataset()
# # print(data.__getitem__(0))
# loader = DataLoader(data, batch_size=1, shuffle=False)
# for a,b,c,d,e,f in loader:
#     print(a.shape,b.shape)
#     print(c.shape,d.shape)
#     print(e.shape,f.shape)
#     exit()