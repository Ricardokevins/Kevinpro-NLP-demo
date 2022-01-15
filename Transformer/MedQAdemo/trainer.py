from torch.utils.data import Dataset
import torch
from util import *
import os
class CharDataset(Dataset):
    def __init__(self):
        self.question,self.answer = readFromPair()
        if os.path.exists('dict.txt') == False:
            self.build_dict(self.question, self.answer)
        else:
            self.load_dict()

    def load_dict(self):
        f = open("dict.txt",'r',encoding = 'utf-8')
        lines = f.readlines()
        self.word2id = {}
        for i in lines:
            data = i.strip().split(' ')
            word = data[0]
            idx = data[1]
            
            self.word2id[word] = int(idx)

        self.id2word = {}
        for i in self.word2id:
            self.id2word[self.word2id[i]] = i
        return 

    def build_dict(self,source,target):
        self.word2id = {'[PAD]':0,'[BOS]':1,'[EOS]':2,'[UNK]':3}
        id = 4
        for s,t in zip(source,target):
            s = s.replace("\n","")
            for i in s:
                if len(i.strip()) != 0:
                    if i not in self.word2id:
                        self.word2id[i] = id 
                        id += 1
            t = t.replace("\n","")
            for i in t:
                if len(i.strip()) != 0:
                    if i not in self.word2id:
                        self.word2id[i] = id 
                        id += 1
            if id > 30000:
                break
        
        self.id2word = {}
        for i in self.word2id:
            self.id2word[self.word2id[i]] = i
        
        f = open('dict.txt','w',encoding='utf-8')
        for i in self.word2id:
            f.write(i + " " + str(self.word2id[i]) + "\n")
        return

    def tokenizer(self,input):
        ids = []
        for i in input:
            if i in self.word2id:
                ids.append(self.word2id[i])
            else:
                ids.append(self.word2id['[UNK]'])
        return ids

    def padding(self,input):
        if len(input) < 200:
            input.append(self.word2id['[PAD]'])
        input = input[:200]
        return input

    def __len__(self):
        return len(self.question)

    def __getitem__(self, idx):
        source_text = self.question[idx]
        source_id = self.tokenizer(source_text)
        source_id = self.padding(source_id)

        target_text = self.answer[idx]
        target_id = self.tokenizer(target_text)

        decode_input = [self.word2id['BOS']] + target_id
        decode_label = target_id + self.word2id['[EOS]']

        return torch.tensor(source_id),torch.tensor(decode_input),torch.tensor(decode_label)


import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False

    ckpt_path = 'novelModel.pth'
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class TransformerTrainer:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model.to(self.device)
            #self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate, betas=config.betas)
        
        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y) in pbar:

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x, y)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        best_loss = float('inf')
        self.tokens = 0 # counter used for learning rate decay
        for epoch in range(config.max_epochs):

            run_epoch('train')
            if self.test_dataset is not None:
                test_loss = run_epoch('test')

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss

            self.save_checkpoint()
            # if self.config.ckpt_path is not None and good_model:
            #     best_loss = test_loss
            #     self.save_checkpoint()

