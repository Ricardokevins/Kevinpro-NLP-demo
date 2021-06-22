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

from dataloader import load_train
from dataloader import load_test
from model import TransformerClasssifier
from model import BiRNN
from model import BiLSTM_Attention1
from model import BiLSTM_Attention2
from model import BertClassifier
from transformers import BertTokenizer
from Attack import FGM
import random

Trainbert = False

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
USE_CUDA = True

def load_eval():
    Trainbert = True
    net = torch.load('teacher.pth')
    net.eval()
    batch_size = 16
    avg_loss = 0
    correct = 0
    total = 0
    iter=0
    crossentropyloss = nn.CrossEntropyLoss()
    with torch.no_grad():
        train_iter = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=True)
    
        for train_text,bert_text,label in train_iter:
            iter += 1
            if train_text.size(0) != batch_size:
                break

            train_text = train_text.reshape(batch_size, -1)
            label = label.reshape(-1)


            if USE_CUDA:
                train_text=train_text.cuda()
                label = label.cuda()
                bert_text = bert_text.cuda()
            logits,attn = net(bert_text)
            loss = crossentropyloss(logits, label)
            avg_loss+=loss.item()     
            #pbar(iter, {'loss': avg_loss/iter})
            _, logits = torch.max(logits, 1)
            # print(logits)
            # print(label)
            # print(correct)
            # print(total)
            correct += logits.data.eq(label.data).cpu().sum()
            total += batch_size
        print("Test Acc : ", correct.numpy().tolist() / total)

from EasyTransformer.util import ProgressBar


num_epochs = 30
batch_size = 128
if Trainbert:
    batch_size = 16


# Choose your model here
net = BiRNN()
# net = BertClassifier()
# net = BiLSTM_Attention1()
# net = BiLSTM_Attention2()
#net = TransformerClasssifier()
net = net.cuda()


def train_with_FGM():
    net.train()
    fgm = FGM(net)
    #optimizer = optim.SGD(net.parameters(), lr=0.01,weight_decay=0.01)
    #optimizer = optim.Adam(net.parameters(), lr=learning_rate,weight_decay=0)
    optimizer = AdamW(net.parameters(),lr = 2e-3, eps = 1e-8)
    #optimizer = AdamW(net.parameters(), lr=learning_rate)
    #train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    pre_loss=1
    crossentropyloss = nn.CrossEntropyLoss()
    total_steps = len(train_iter)*num_epochs
    print("----total step: ",total_steps,"----")
    print("----warmup step: ", int(total_steps * 0.2), "----")
    
    best_acc = 0
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = int(total_steps*0.15), num_training_steps = total_steps)
    for epoch in range(num_epochs):
        correct = 0
        total=0
        iter = 0
        pbar = ProgressBar(n_total=len(train_iter), desc='Training')
        net.train()
        avg_loss = 0
        for train_text,bert_text,label in train_iter:
            iter += 1
            if train_text.size(0) != batch_size:
                break

            train_text = train_text.reshape(batch_size, -1)
            label = label.reshape(-1)


            if USE_CUDA:
                train_text=train_text.cuda()
                label = label.cuda()

            logits, attn = net(train_text)
           
            loss = crossentropyloss(logits, label)
            
            loss.backward()
            
            avg_loss += loss.item()
            fgm.attack()
            logits, attn = net(train_text)
            loss_adv = crossentropyloss(logits, label)
            loss_adv.backward() 
            fgm.restore()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            pbar(iter, {'loss': avg_loss/iter})
            _, logits = torch.max(logits, 1)

            correct += logits.data.eq(label.data).cpu().sum()
            total += batch_size
        loss=loss.detach().cpu()
        #print("\nepoch ", str(epoch)," loss: ", loss.mean().numpy().tolist(),"Acc:", correct.numpy().tolist()/total)
        cur_acc = test()
        if best_acc < cur_acc:
            best_acc = cur_acc
            torch.save(net, 'model1.pth') 
    
    print(best_acc)
    return

def test():
    net.eval()
    batch_size = 16
    avg_loss = 0
    correct = 0
    total = 0
    iter=0
    crossentropyloss = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        train_iter = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=True)
        pbar = ProgressBar(n_total=len(train_iter), desc='Testing')
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
            if Trainbert:
                logits,attn = net(bert_sent)
            else:
                logits,attn = net(train_text)
        
            loss = crossentropyloss(logits, label)
            avg_loss+=loss.item()     
            #pbar(iter, {'loss': avg_loss/iter})
            _, logits = torch.max(logits, 1)

            correct += logits.data.eq(label.data).cpu().sum()
            total += batch_size
        print("Test Acc : ", correct.numpy().tolist() / total)
        return correct.numpy().tolist() / total

def train_Kd():
    load_eval()
    teach_model = BertClassifier()
    teach_model=torch.load('./teacher.pth')
    teach_model.eval()
    teach_model = teach_model.cuda()
    
    optimizer = AdamW(net.parameters(),lr = 2e-3, eps = 1e-8)
    if Trainbert:
        optimizer = AdamW(net.parameters(),lr = 2e-5, eps = 1e-8)
    #optimizer = AdamW(net.parameters(), lr=learning_rate)
    #train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    pre_loss=1
    alpha = 0.3
    criterion2 = nn.MSELoss()
    crossentropyloss = nn.CrossEntropyLoss()
    total_steps = len(train_iter)*num_epochs
    print("----total step: ",total_steps,"----")
    print("----warmup step: ", int(total_steps * 0.2), "----")
    best_acc=0
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = int(total_steps*0.15), num_training_steps = total_steps)
    
    for epoch in range(num_epochs):
        correct = 0
        total=0
        iter = 0
        pbar = ProgressBar(n_total=len(train_iter), desc='Training')
        net.train()
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

            with torch.no_grad():
                teacher_outputs,attn = teach_model(bert_sent)
            if Trainbert:
                logits,attn = net(bert_sent)
            else:
                logits,attn = net(train_text)
            
            T = 2
            outputs_S = F.softmax(logits/T,dim=1)
            outputs_T = F.softmax(teacher_outputs/T,dim=1)

            loss2 = criterion2(outputs_S,outputs_T)*T*T     
            loss = crossentropyloss(logits, label)
            loss = loss*(1-alpha) + loss2*alpha

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            avg_loss+=loss.item()     
            pbar(iter, {'loss': avg_loss/iter})
            _, logits = torch.max(logits, 1)
            # print(logits)
            # print(label)
            # print(correct)
            # print(total)
            correct += logits.data.eq(label.data).cpu().sum()
            total += batch_size
        loss=loss.detach().cpu()
        #print("\nepoch ", str(epoch)," loss: ", loss.mean().numpy().tolist(),"Acc:", correct.numpy().tolist()/total)
        cur_acc = test()
        if best_acc < cur_acc:
            best_acc = cur_acc
            print("saved Best ACC: ",best_acc)
            torch.save(net, 'model.pth') 
    
    print(best_acc)
    return

def train():
    net.train()
    #optimizer = optim.SGD(net.parameters(), lr=0.01,weight_decay=0.01)
    #optimizer = optim.Adam(net.parameters(), lr=learning_rate,weight_decay=0)
    
    optimizer = AdamW(net.parameters(),lr = 2e-3, eps = 1e-8)
    if Trainbert:
        optimizer = AdamW(net.parameters(),lr = 2e-5, eps = 1e-8)
    #optimizer = AdamW(net.parameters(), lr=learning_rate)
    #train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    pre_loss=1
    crossentropyloss = nn.CrossEntropyLoss()
    total_steps = len(train_iter)*num_epochs
    print("----total step: ",total_steps,"----")
    print("----warmup step: ", int(total_steps * 0.2), "----")
    best_acc=0
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = int(total_steps*0.15), num_training_steps = total_steps)
    for epoch in range(num_epochs):
        correct = 0
        total=0
        iter = 0
        pbar = ProgressBar(n_total=len(train_iter), desc='Training')
        net.train()
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
                
            if Trainbert:
                logits,attn = net(bert_sent)
            else:
                logits,attn = net(train_text)
            loss = crossentropyloss(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            avg_loss+=loss.item()     
            pbar(iter, {'loss': avg_loss/iter})
            _, logits = torch.max(logits, 1)
            # print(logits)
            # print(label)
            # print(correct)
            # print(total)
            correct += logits.data.eq(label.data).cpu().sum()
            total += batch_size
        loss=loss.detach().cpu()
        #print("\nepoch ", str(epoch)," loss: ", loss.mean().numpy().tolist(),"Acc:", correct.numpy().tolist()/total)
        cur_acc = test()
        if best_acc < cur_acc:
            best_acc = cur_acc
            print("saved Best ACC: ",best_acc)
            torch.save(net, 'model.pth') 
    
    print(best_acc)
    return
    
# base 0.71875
# Attention1 0.7777777777777778
# Attention2 0.7430555555555556
# Transformer 0.7708333333333334


# 0.7048611111111112
# 0.78125
# 0.7743055555555556
# 0.71875
#train_with_FGM()

# NOVA
# trainer()
# Base 0.7465277777777778
# Attention1 0.8159722222222222
# Attention2 0.7881944444444444
# Transformer 0.7916666666666666
# BERT 0.8645833333333334


Trainbert = False
#train()
#train_Kd()
train_with_FGM()

# Teacher = 0.8819444444444444
# base = 0.7881944444444444
# Attention1 = 0.8194444444444444
# Attention2 = 0.8159722222222222
# Transformer = 0.7916666666666666