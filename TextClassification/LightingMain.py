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
import torch

from dataloader import load_train
from dataloader import load_test
from dataloader import get_dict_size
from model import TransformerClasssifier
from model import BiRNN
from model import BiLSTM_Attention1
from model import BiLSTM_Attention2
from model import BertClassifier
from model import RDrop
from transformers import BertTokenizer
from pytorch_lightning.lite import LightningLite
from Attack import FGM
import random

from config import *
import warnings

warnings.filterwarnings('ignore')
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
setup_seed(44)

train_samples = load_train()
test_samples = load_test()

def convert2dataset_origin(text,bert_sent,label):
    text=torch.tensor(text)
    print("Text shape",text.shape)
    bert_sent=torch.tensor(bert_sent)
    label = torch.tensor(label)
    train_dataset = torch.utils.data.TensorDataset(text,bert_sent,label)
    return train_dataset

train_dataset = convert2dataset_origin(train_samples[0], train_samples[1],train_samples[2])
test_dataset = convert2dataset_origin(test_samples[0], test_samples[1],test_samples[2])





class Lite(LightningLite):
    def run(self, args):
        model = BiLSTM_Attention1()
        model = model.cuda()
        optimizer = AdamW(model.parameters(),lr = 2e-4, eps = 1e-8)
        model, optimizer = self.setup(model, optimizer)  # Scale your model / optimizers
        train_iter = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
        dataloader = self.setup_dataloaders(train_iter)  # Scale your dataloaders

        model.train()
        pre_loss=1
        crossentropyloss = nn.CrossEntropyLoss()
        if mode == 'train_with_RDrop':
            KL_loss_function = nn.KLDivLoss()
        #total_steps = len(train_iter)*num_epochs
        #print("----total step: ",total_steps,"----")
        #print("----warmup step: ", int(total_steps * 0.2), "----")
        best_acc=0
        for epoch in range(num_epochs):
            print("EPOCH {}".format(epoch))
            correct = 0
            total=0
            iter = 0
            from EasyTransformer.util import ProgressBar
            pbar = ProgressBar(n_total=len(train_iter), desc='Training')
            model.train()
            avg_loss = 0
            for train_text,bert_sent,label in train_iter:
                iter += 1
                if train_text.size(0) != batch_size:
                    break

                train_text = train_text.reshape(batch_size, -1)
                label = label.reshape(-1)


                if USE_CUDA:
                    train_text=train_text.cuda()
                    label = label.cuda()
                    bert_sent = bert_sent.cuda()
                    

                logits,attn = model(train_text)


                loss = crossentropyloss(logits, label)

                optimizer.zero_grad()
                self.backward(loss)
                optimizer.step()
   
                avg_loss+=loss.item()     
                pbar(iter, {'loss': avg_loss/iter})
                _, logits = torch.max(logits, 1)
                correct += logits.data.eq(label.data).cpu().sum()
                total += batch_size
            loss=loss.detach().cpu()
            
        print(best_acc)
        return


Lite(devices="auto", accelerator="auto", precision=16).run(10)
#Lite(strategy="deepspeed", devices=1, accelerator="gpu", precision=16).run(10)
#Lite(devices=1, accelerator="gpu", precision=32).run(10)