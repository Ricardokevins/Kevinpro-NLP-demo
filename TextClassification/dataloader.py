import transformer
train_data='trains.txt'
test_data='tests.txt'

import csv


# 读取csv至字典
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
    
def build_tokenizer():
    model = transformer.Transformer(max_length=max_length)
    text,_=get_dataset(train_data)
    tokenizer = model.get_tokenzier(text)
    return tokenizer

tokenzier = build_tokenizer()

def load_train():
    sents = []
    text,labels = get_dataset(train_data)

    for i in text:
        sents.append(tokenzier.encode(i))

    return sents,labels

def load_test():
    sents = []
    text,labels = get_dataset(test_data)

    for i in text:
        sents.append(tokenzier.encode(i))

    return sents, labels
    