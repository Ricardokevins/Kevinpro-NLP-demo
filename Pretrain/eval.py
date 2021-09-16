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

from dataloader import load_train_data
from dataloader import load_test_data
from model import TransformerClasssifier
from model import BiRNN

from transformers import BertTokenizer
import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
setup_seed(44)

from dataloader import get_dict_size
word_size = get_dict_size()

def sample(source,target,label):
    print(source)
    print(target)
    print(label)
#sample(source,target,label)
def get_dataset(source,target,label):
    source = torch.tensor(source)
    target = torch.tensor(target)
    label = torch.tensor(label)
    train_dataset = torch.utils.data.TensorDataset(source,target,label)

    return train_dataset



USE_CUDA = True

from EasyTransformer.util import ProgressBar


num_epochs = 30
batch_size = 128



def test(net):
    net.eval()
    batch_size = 2
    avg_loss = 0
    correct = 0
    total = 0
    iter=0
    crossentropyloss = nn.CrossEntropyLoss()
    source,target,label = load_test_data()
    #sample(source,target,label)
    test_dataset = get_dataset(source,target,label)
    with torch.no_grad():
        train_iter = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=True)
        pbar = ProgressBar(n_total=len(train_iter), desc='Testing')
        for source,target,label in train_iter:
            iter += 1
            if source.size(0) != batch_size:
                break

            source = source.reshape(batch_size, -1)
            target = target.reshape(batch_size,-1)

            label = label.reshape(batch_size,-1)


            if USE_CUDA:
                source=source.cuda()
                target=target.cuda()
                label = label.cuda()
        
            logits,attn = net(source)

            #print(logits.shape)
            #print(target.shape)
            active_loss = label.reshape(-1) == 1

            active_logits = logits.reshape(-1,word_size)[active_loss]
            active_labels = target.reshape(-1)[active_loss]
            #print(active_logits.shape)
            #print(active_labels.shape)
            #loss = loss_fct(active_logits, active_labels)
            loss = crossentropyloss(active_logits,active_labels)    
            #pbar(iter, {'loss': avg_loss/iter})
            _, active_logits = torch.max(active_logits, 1)
            sample(source,target,label)
            print(active_loss)
            print(active_labels)
            print(active_logits)
            exit()
            correct += active_logits.data.eq(active_labels.data).cpu().sum()
            total += active_labels.shape[0]
        loss=loss.detach().cpu()
        print("Test Acc : ", " loss: ", loss.mean().numpy().tolist(),"Acc:", correct.numpy().tolist()/total)
        return correct.numpy().tolist() / total

model=torch.load('model.pth')
model = model.cuda()
test(model)

