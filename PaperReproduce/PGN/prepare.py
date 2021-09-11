from dataloader import Tokenizer
from dataloader import read_data_from_json
from config import train_data_path

def read_data_from_json(filename):
    with open("filename", 'r', encoding='utf-8') as load_f:
        temp = load_f.readlines()
        for line in temp:
            dic = json.loads(line)
            article.append(dic['content'])
            summary.append(dic['summary'])
    return article,summary


def build_vocab():
    article,summary = read_data_from_json(train_data_path)