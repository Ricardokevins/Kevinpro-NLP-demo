import torch
import  transformer
max_length = 64
def get_dataset(corpur_path):
    texts = []
    labels = []
    with open(corpur_path) as f:
        li = []
        while True:
            content = f.readline().replace('\n', '')
            if not content:              #为空行，表示取完一次数据（一次的数据保存在li中）
                if not li:               #如果列表也为空，则表示数据读完，结束循环
                    break
                label = li[0][10]
                text = li[1][6:-7]
                texts.append(text)
                labels.append(int(label))
                li = []
            else:
                li.append(content)       #["<Polarity>标签</Polarity>", "<text>句子内容</text>"]
    return texts, labels


def bulid_corpus(train_data):
    texts, labels = get_dataset(train_data)
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

corpus = bulid_corpus('trains.txt')
tokenzier = build_tokenizer(corpus)

def get_dict_size():
    return len(tokenzier.word2idx)

def load_train_data(samples_num):
    corpus = bulid_corpus('trains.txt')
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
    corpus = bulid_corpus('tests.txt')
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
