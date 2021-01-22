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
from Attack import FGM
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

train_samples = load_train()
test_samples = load_test()

def convert2dataset_origin(text,label):
    text=torch.tensor(text)
    label = torch.tensor(label)
    train_dataset = torch.utils.data.TensorDataset(text,label)
    return train_dataset

train_dataset = convert2dataset_origin(train_samples[0], train_samples[1])
test_dataset = convert2dataset_origin(test_samples[0], test_samples[1])

from util import ProgressBar
USE_CUDA = True

num_epochs = 20
batch_size = 64
net = BiRNN()
#net = TransformerClasssifier()
net = net.cuda()

Attack_with_FGM = True
def train_with_FGM():
    net.train()
    fgm = FGM(net)
    #optimizer = optim.SGD(net.parameters(), lr=0.01,weight_decay=0.01)
    #optimizer = optim.Adam(net.parameters(), lr=learning_rate,weight_decay=0)
    optimizer = AdamW(net.parameters(),lr = 2e-5, eps = 1e-8)
    #optimizer = AdamW(net.parameters(), lr=learning_rate)
    #train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    pre_loss=1
    crossentropyloss = nn.CrossEntropyLoss()
    total_steps = len(train_iter)*num_epochs
    print("----total step: ",total_steps,"----")
    print("----warmup step: ",int(total_steps*0.2),"----")
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = int(total_steps*0.15), num_training_steps = total_steps)
    for epoch in range(num_epochs):
        correct = 0
        total=0
        iter = 0
        pbar = ProgressBar(n_total=len(train_iter), desc='Training')
        net.train()
        avg_loss = 0
        for train_text,label in train_iter:
            iter += 1
            if train_text.size(0) != batch_size:
                break

            train_text = train_text.reshape(batch_size, -1)
            label = label.reshape(-1)


            if USE_CUDA:
                train_text=train_text.cuda()
                label = label.cuda()

            logits = net(train_text)
            loss = crossentropyloss(logits, label)
            
            loss.backward()
            
            avg_loss += loss.item()
            fgm.attack()
            logits = net(train_text)
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
        print("\nepoch ", str(epoch)," loss: ", loss.mean().numpy().tolist(),"right", correct.numpy().tolist(), "total", total, "Acc:", correct.numpy().tolist()/total)
        test()
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
        for train_text,label in train_iter:
            iter += 1
            if train_text.size(0) != batch_size:
                break

            train_text = train_text.reshape(batch_size, -1)
            label = label.reshape(-1)


            if USE_CUDA:
                train_text=train_text.cuda()
                label = label.cuda()

            logits = net(train_text)
            loss = crossentropyloss(logits, label)
            avg_loss+=loss.item()     
            pbar(iter, {'loss': avg_loss/iter})
            _, logits = torch.max(logits, 1)
            # print(logits)
            # print(label)
            # print(correct)
            # print(total)
            correct += logits.data.eq(label.data).cpu().sum()
            total += batch_size
        print("Test Acc : ", correct.numpy().tolist() / total)


def trainer():
    net.train()
    #optimizer = optim.SGD(net.parameters(), lr=0.01,weight_decay=0.01)
    #optimizer = optim.Adam(net.parameters(), lr=learning_rate,weight_decay=0)
    optimizer = AdamW(net.parameters(),lr = 2e-5, eps = 1e-8)
    #optimizer = AdamW(net.parameters(), lr=learning_rate)
    #train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    pre_loss=1
    crossentropyloss = nn.CrossEntropyLoss()
    total_steps = len(train_iter)*num_epochs
    print("----total step: ",total_steps,"----")
    print("----warmup step: ",int(total_steps*0.2),"----")
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = int(total_steps*0.15), num_training_steps = total_steps)
    for epoch in range(num_epochs):
        correct = 0
        total=0
        iter = 0
        pbar = ProgressBar(n_total=len(train_iter), desc='Training')
        net.train()
        avg_loss = 0
        for train_text,label in train_iter:
            iter += 1
            if train_text.size(0) != batch_size:
                break

            train_text = train_text.reshape(batch_size, -1)
            label = label.reshape(-1)


            if USE_CUDA:
                train_text=train_text.cuda()
                label = label.cuda()

            logits = net(train_text)
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
        print("\nepoch ", str(epoch)," loss: ", loss.mean().numpy().tolist(),"right", correct.numpy().tolist(), "total", total, "Acc:", correct.numpy().tolist()/total)
        test()
    return

#trainer()
train_with_FGM()