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
corpus = bulid_corpus('corpus.txt',1)
tokenzier = build_tokenizer(corpus)
from dataloader import get_dict_size
word_size = get_dict_size()
print("wprd+size",word_size)
def propossess_text(text):
    label = []
    for i in text.split(" "):
        if i == '[MASK]':
            label.append(1)
        else:
            label.append(0)
    
    max_length = 80
    while len(label)<max_length:
        label.append(0)
    label = label[:max_length]
    return label

def fill_blank():
    text = "I love it and it is good."
    input_text = "I love it and it is [MASK] ."
    #text = "i will never visit this restaurant again."
    #input_text = "i will never [MASK] this restaurant again. "
    #text = "I have bought several of the Vitality canned dog food products and have found them all to be of good quality."
    #input_text = "I have bought several of the Vitality canned dog food products and [MASK] found them all to be of [MASK] quality."
    label = propossess_text(input_text)
    input_id = tokenzier.encode(input_text)
    print("model wordsize :",word_size)
    model = TransformerClasssifier(word_size)
    model=torch.load('model.pth')
    model = model.cuda()
    input_id = torch.tensor(input_id).reshape(1,-1).cuda()
    logits,attn = model(input_id)
    
    label = torch.tensor(label)
    active_loss = label.reshape(-1) == 1
    #print(logits.shape)
    active_logits = logits.reshape(-1,word_size)[active_loss]

    _, active_logits = torch.topk(active_logits, k = 20 , dim=1) 
    #_, active_logits = torch.max(active_logits, 1)

    predict = active_logits.cpu().numpy().tolist()
    #print(predict)
    
    print(text)
    print(input_text)
    for i in predict:
        for j in i:
            print("The [MASK] Blank Answer: ",tokenzier.idx2word[j])
        print('\n')


fill_blank()
