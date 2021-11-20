from EasyTransformer import transformer
from config import *
from transformers import BertTokenizer
import csv

train_data='../data/trains.txt'
test_data='../data/tests.txt'
Prompt_pattern = "It's a [MASK] restaurant. <text a>"

def apply_pattern(input_sentence):
    result = Prompt_pattern.replace("<text a>",input_sentence)
    return result

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
                text = apply_pattern(text)
                texts.append(text)
                labels.append(int(label))
                li = []
            else:
                li.append(content)       #["<Polarity>标签</Polarity>", "<text>句子内容</text>"]

    return texts, labels
    
bert_tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

def get_dict_size():
    return tokenzier.get_size()
    
def load_train():
    bert_sent = []
    masks = []
    text,labels = get_dataset(train_data)
    from tqdm import tqdm 
    for i in tqdm(text):
        indexed_tokens = bert_tokenizer.encode(i, add_special_tokens=True)
        while len(indexed_tokens) <  max_length:
            indexed_tokens.append(0)  # 0 is id for [PAD]
        indexed_tokens = indexed_tokens[: max_length]
        mask = [0 for i in range(max_length)]
        index = indexed_tokens.index(103)
        mask[index] = 1
        masks.append(mask)
        bert_sent.append(indexed_tokens)

    return bert_sent,masks,labels

def load_test():
    bert_sent = []
    masks = []
    text,labels = get_dataset(test_data)
    from tqdm import tqdm 
    for i in tqdm(text):
        indexed_tokens = bert_tokenizer.encode(i, add_special_tokens=True)
        while len(indexed_tokens) <  max_length:
            indexed_tokens.append(0)  # 0 is id for [PAD]
        indexed_tokens = indexed_tokens[: max_length]
        mask = [0 for i in range(max_length)]
        index = indexed_tokens.index(103)
        mask[index] = 1
        masks.append(mask)
        bert_sent.append(indexed_tokens)

    return bert_sent,masks,labels

load_train()
