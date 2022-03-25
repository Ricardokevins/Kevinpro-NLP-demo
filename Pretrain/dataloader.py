import torch
import  transformer
max_length = 10
max_sample = 1000

def get_dataset(corpur_path,max_num):
    texts = []
    labels = []
    with open(corpur_path,encoding='utf-8') as f:
        li = []
        while True:
            content = f.readline().replace('\n', '')
            texts.append(content)
            if len(texts) > max_num:
                break
    return texts, labels


def bulid_corpus(train_data,max_num):
    texts, labels = get_dataset(train_data,max_num)
    return texts

def build_tokenizer(text):
    model = transformer.Transformer(max_length=max_length)
    tokenizer = model.get_tokenzier(text)
    return tokenizer

import random
import copy
def random_word(text,tokenzier):
    tokens = copy.deepcopy(text)
    output_label = []
    for i, token in enumerate(tokens):
            if token == 0:
                break
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = 4 #MASK ID

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(7,len(tokenzier.word2idx))

                # 10% randomly change token to current token
                else:
                    pass

                output_label.append(1)

            else:
                output_label.append(0)

    return tokens,output_label

corpus = bulid_corpus('corpus.txt',5e3)
tokenzier = build_tokenizer(corpus)

def get_dict_size():
    return len(tokenzier.word2idx)

def load_train_data():
    corpus = bulid_corpus('corpus.txt',max_sample)
    source = []
    target = []
    labels = []
    while 1:
        for i in corpus:
            target_id = tokenzier.encode(i)
            source_id,label = random_word(target_id,tokenzier)
            #print(type(tokenzier.encode(text)))
            #assert(len(tokenzier.encode(text))==len(tokenzier.encode(i)))
            source.append(source_id)
            target.append(target_id)
            while len(label)<max_length:
                label.append(0)
            label = label[:max_length]
            labels.append(label)
            #assert(len(label) == max_length)
            if len(source) >= (max_sample):
                return source,target,labels


def load_test_data():
    corpus = bulid_corpus('corpus.txt',max_sample/10)
    source = []
    target = []
    labels = []
    while 1:
        for i in corpus:
            target_id = tokenzier.encode(i)
            source_id,label = random_word(target_id,tokenzier)
            #print(type(tokenzier.encode(text)))
            #assert(len(tokenzier.encode(text))==len(tokenzier.encode(i)))
            source.append(source_id)
            target.append(target_id)
            while len(label)<max_length:
                label.append(0)
            label = label[:max_length]
            labels.append(label)
            #assert(len(label) == max_length)
            if len(source) >= (max_sample/10):
                return source,target,labels
