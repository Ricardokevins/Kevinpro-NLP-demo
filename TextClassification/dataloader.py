from EasyTransformer import transformer
train_data='./data/trains.txt'
test_data='./data/tests.txt'
from transformers import BertTokenizer
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
    text, _ = get_dataset(train_data)
    tokenizer = model.get_base_tokenzier(text)
    #tokenizer = model.get_BPE_tokenizer(text)
    #tokenizer = model.get_Char_tokenizer(text)
    return tokenizer

tokenzier = build_tokenizer()
bert_tokenizer = BertTokenizer.from_pretrained('../../../PTM/bert-base-chinese')

def get_dict_size():
    return tokenzier.get_size()
    
def load_train():
    sents = []
    bert_sent = []
    text,labels = get_dataset(train_data)
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

    for i in text:
        sents.append(tokenzier.encode(i))
        indexed_tokens = bert_tokenizer.encode(i, add_special_tokens=True)
        while len(indexed_tokens) <  max_length:
            indexed_tokens.append(0)  # 0 is id for [PAD]
        indexed_tokens = indexed_tokens[: max_length]
        bert_sent.append(indexed_tokens)

    return sents,bert_sent,labels
    