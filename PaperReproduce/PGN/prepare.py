from dataloader import Tokenizer
from dataloader import read_data_from_json
from config import train_data_path
from config import vocab_size
from tqdm import tqdm 
def build_vocab():
    t = Tokenizer(False)
    article,summary = read_data_from_json(train_data_path)
    for i,j in zip(article,summary):
        t.add_sentence(i)
        t.add_sentence(j)
        if t.cur_word>=vocab_size:
            break
        print(t.cur_word)
    t.export()

build_vocab()
