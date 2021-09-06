import torch
import jieba
from tqdm import tqdm
from EasyTransformer import transformer
from transformers import BertTokenizer
import csv

max_length = 256
train_data='./data/cntrain.txt'
test_data='./data/cntest.txt'

def pre_process(corpur_path):
    f = open(corpur_path,'r',encoding='utf-8')
    lines = f.readlines()
    label = []
    corpus = []
    for i in tqdm(lines):
        temp = i.split('	')
        temp2 = temp[1].strip()
        temp2 = temp2.replace(' ','')
        corpus.append(" ".join(jieba.cut(temp2, cut_all=False)))
        label.append(temp[0])
    f = open("./data/cntest.txt",'w',encoding='utf-8')
    for i,j in zip(label,corpus):
        f.write(i+"\t"+j+'\n')
    f.close()


def get_category():
    labels = []
    f = open(train_data,'r',encoding='utf-8')
    lines = f.readlines()
    for i in lines:
        l = i.strip().split('\t')[0]
        labels.append(l)
    id2label = {}
    label2id = {}
    for index,i in enumerate(list(set(labels))):
        id2label[index]= i
        label2id[i] = index

    return id2label, label2id

id2label, label2id = get_category()

def get_dataset(corpur_path):
    texts = []
    labels = []
    f = open(corpur_path,'r',encoding='utf-8')
    lines = f.readlines()
    for i in lines:
        t = i.strip().split('\t')[1]
        l = label2id[i.strip().split('\t')[0]]
        texts.append(t)
        labels.append(l)
    return texts, labels

def build_tokenizer():
    model = transformer.Transformer(max_length=max_length)
    text, _ = get_dataset(train_data)
    tokenizer = model.get_base_tokenzier(text)
    return tokenizer


tokenzier = build_tokenizer()
bert_tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')

def get_dict_size():
    return tokenzier.get_size()
    
def load_train():
    sents = []
    bert_sent = []
    text,labels = get_dataset(train_data)
    text = text[:10000]
    labels = labels[:10000]
    from tqdm import tqdm 
    for i in tqdm(text):
        sents.append(tokenzier.encode(i))
        indexed_tokens = bert_tokenizer.encode(i, add_special_tokens=True)
        while len(indexed_tokens) <  max_length:
            indexed_tokens.append(0)  # 0 is id for [PAD]
        indexed_tokens = indexed_tokens[: max_length]
        bert_sent.append(indexed_tokens)
    
    print(text[0])
    print(sents[0])
    return sents,bert_sent,labels

def load_test():
    sents = []
    bert_sent = []
    text,labels = get_dataset(test_data)
    text = text[:1000]
    labels = labels[:1000]
    for i in text:
        sents.append(tokenzier.encode(i))
        indexed_tokens = bert_tokenizer.encode(i, add_special_tokens=True)
        while len(indexed_tokens) <  max_length:
            indexed_tokens.append(0)  # 0 is id for [PAD]
        indexed_tokens = indexed_tokens[: max_length]
        bert_sent.append(indexed_tokens)

    return sents,bert_sent,labels
    