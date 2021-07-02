import numpy as np
import time
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.functional
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import torch
import  transformer
from dataloader import load_train_data
from dataloader import load_test_data
from model import TransformerClasssifier
from model import BiRNN
from model import BiLSTM_Attention1
from model import BiLSTM_Attention2
from model import BertClassifier
from transformers import BertTokenizer
import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
setup_seed(44)

from dataloader import bulid_corpus,build_tokenizer
corpus = bulid_corpus('trains.txt')
tokenzier = build_tokenizer(corpus)
from dataloader import get_dict_size
word_size = get_dict_size()

def propossess_text(text):
    label = []
    for i in text.split(" "):
        if i == '[MASK]':
            label.append(1)
        else:
            label.append(0)
    
    max_length = 64
    while len(label)<max_length:
        label.append(0)
    label = label[:max_length]
    return label

def fill_blank():
    text = "i will never visit this restaurant again."
    input_text = "i will [MASK] visit this restaurant [MASK]"
    # text = "i love the little pie company as much as anyone else who has written reviews"
    # input_text = "i love the little pie company as [MASK] as anyone else who [MASK] written reviews"
    label = propossess_text(input_text)
    input_id = tokenzier.encode(text)
    model = TransformerClasssifier(word_size)
    model=torch.load('test.pth')
    model = model.cuda()
    input_id = torch.tensor(input_id).reshape(1,-1).cuda()
    logits,attn = model(input_id)
    
    label = torch.tensor(label)
    active_loss = label.reshape(-1) == 1
    active_logits = logits.reshape(-1,word_size)[active_loss]
    _, active_logits = torch.max(active_logits, 1)
    predict = active_logits.cpu().numpy().tolist()
    print(text)
    print(input_text)
    for i in predict:
        print("[MASK] Blank Answer: ",tokenzier.idx2word[i])


fill_blank()
