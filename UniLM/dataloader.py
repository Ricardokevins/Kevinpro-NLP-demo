from transformers import BertTokenizer
from tqdm import tqdm
bert_tokenizer = BertTokenizer.from_pretrained('../../PLM/bert-base-chinese')
def read_paired_data(path):
    f_source = open(source_path,'r',encoding = 'utf-8')
    source = f.readlines()
    source = [i.strip() for i in source]
    f_target = open(target_path,'r',encoding = 'utf-8')
    target = f.readlines()
    target = [i.strip() for i in target]
    return source, target

def language_model_format():
    f = open('xz.txt','r',encoding = 'utf-8')
    lines = f.readlines()
    filterd = []
    for i in tqdm(lines):
        if (i.strip()) !=0:
            filterd.append(i)
    print(filterd[0])

language_model_format()

def get_train_set():
    exit()