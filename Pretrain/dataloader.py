import torch
import  transformer
max_length = 80
max_sample = 1e4

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
def random_word(text,tokenzier):
    tokens = text.split(' ')
    output_label = []
    for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = '[MASK]'

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = tokenzier.idx2word[random.randrange(len(tokenzier.word2idx))]

                # 10% randomly change token to current token
                else:
                    pass

                output_label.append(1)

            else:
                output_label.append(0)

    return " ".join(tokens),output_label

corpus = bulid_corpus('corpus.txt',5e3)
tokenzier = build_tokenizer(corpus)

def get_dict_size():
    return len(tokenzier.word2idx)

def load_train_data(samples_num):
    corpus = bulid_corpus('corpus.txt',samples_num/2)
    source = []
    target = []
    labels = []
    while 1:
        for i in corpus:
            text,label = random_word(i,tokenzier)
            source.append(tokenzier.encode(text))
            target.append(tokenzier.encode(i))
            while len(label)<max_length:
                label.append(0)
            label = label[:max_length]
            assert(len(label) == max_length)
            labels.append(label)
            if len(source) >= samples_num:
                return source,target,labels

def load_test_data(samples_num):
    corpus = bulid_corpus('corpus.txt',samples_num/2)
    source = []
    target = []
    labels = []
    while 1:
        for i in corpus:
            text,label = random_word(i,tokenzier)
            source.append(tokenzier.encode(text))
            target.append(tokenzier.encode(i))
            while len(label)<max_length:
                label.append(0)
            label = label[:max_length]
            labels.append(label)
            assert(len(label) == max_length)
            if len(source) >= samples_num:
                return source,target,labels
