from data import SumDataset
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import FileStorageObserver
from torch import optim

from EasyTransformer.util import ProgressBar
from torch.nn.utils import clip_grad_norm_
from transformers import AdamW
from model import EncoderRNN
from model import LuongAttnDecoderRNN
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
import os
import time

import taskConfig as con

ex = Experiment('Seq2Seq', save_git_info=False)

Traing_flag = False
ex.captured_out_filter = apply_backspaces_and_linefeeds
obv = MongoObserver(url="localhost", port=27017, db_name="PgnSum")

if Traing_flag:
    ex.observers.append(FileStorageObserver('Run'))
    ex.observers.append(obv)


@ex.config
def config():
    batch_size = con.batch_size
    epochs = con.epoch
    cuda_device = [0]
    optimizer = {'choose': con.optim, 'lr': con.lr, 'eps': con.eps}
    #optimizer = {'choose': 'SGD', 'lr': 0.01}
    loss_func = 'CE'
    attn_model = con.attn_model
    vocabSize = con.max_vocab
    hidden_dim = con.hidden_size

    dropout = con.dropout
    encoder_n_layers = con.encoder_n_layers
    decoder_n_layers = con.decoder_n_layers

    decoder_lr_ratio = con.decoder_learning_ratio
    teacher_forcing_ratio = con.teacher_forcing_ratio

    clip = con.clip

def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    #print(inp.shape,target.shape)
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.cuda()
    return loss, nTotal.item()

class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * 2
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        print(all_tokens)
        print(all_scores)
        return all_tokens, all_scores

class Trainer:
    def __init__(self, config, runner):
        self.runner = runner

        self.batch_size = config["batch_size"]
        self.epoch = config['epochs']
        self.device_ids = config["cuda_device"]
        self.vocabSize = config["vocabSize"]
        self.hidden_dim = config["hidden_dim"]
        self.attn_model = config["attn_model"]
        self.dropout = config['dropout']
        self.encoder_n_layers = config['encoder_n_layers']
        self.decoder_n_layers = config['decoder_n_layers']
        self.decoder_lr_ratio = config['decoder_lr_ratio']
        self.teacher_forcing_ratio = config['teacher_forcing_ratio']

        self.clip = config['clip']

        print("----- get model ----")
        self.encoder, self.decoder,self.embedding = self.get_model()

        if config['optimizer']['choose'] == "AdamW":
            self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=config['optimizer']['lr'])
            self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=config['optimizer']['lr'] * self.decoder_lr_ratio)
    


        if config['loss_func'] == "CE":
            self.criterion = nn.CrossEntropyLoss()

        data = SumDataset()
        self.dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=True,drop_last=True)

        # self.model=nn.DataParallel(self.model,device_ids=self.device_ids)

    def get_model(self):
        
        embedding = nn.Embedding(self.vocabSize, self.hidden_dim).cuda()
        encoder = EncoderRNN(self.hidden_dim, embedding, self.encoder_n_layers, self.dropout).cuda()
        decoder = LuongAttnDecoderRNN(self.attn_model, embedding, self.hidden_dim, self.vocabSize, self.decoder_n_layers, self.dropout).cuda()
        return encoder,decoder,embedding

    def train(self):
        import random
        print("---- Start Training ----")
        log_freq = int(len(self.dataloader) / 10)

        pbar = ProgressBar(n_total=len(self.dataloader), desc='Training')

        best_loss = 1000000

        for epo in range(self.epoch):
            self.encoder.train()
            self.decoder.train()
            count =0 
            for source_sent, target_sent, source_length, target_length, source_mask, target_mask in self.dataloader:
                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()
                source_sent, target_sent, source_length, target_length, source_mask, target_mask =source_sent.cuda(), target_sent.cuda(), source_length.cuda(), target_length.cuda(), source_mask.cuda(), target_mask.cuda()
                loss = 0
                print_losses = []
                n_totals = 0


                encoder_outputs, encoder_hidden = self.encoder(source_sent, source_length)
                decoder_input = torch.LongTensor([[2 for _ in range(self.batch_size)]])
                decoder_input = decoder_input.cuda()
                decoder_hidden = encoder_hidden[: self.decoder.n_layers]
                use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
                #use_teacher_forcing = True
                max_target_len = torch.max(target_length).item()
                target_sent = target_sent.transpose(0, 1)
                target_mask =target_mask.transpose(0, 1)

                if use_teacher_forcing:
                    for t in range(max_target_len):
                        decoder_output, decoder_hidden = self.decoder(
                            decoder_input, decoder_hidden, encoder_outputs
                        )
                        # Teacher forcing: next input is current target
                        decoder_input = target_sent[t].view(1, -1)
                        # Calculate and accumulate loss
                        
                        mask_loss, nTotal = maskNLLLoss(decoder_output, target_sent[t], target_mask[t])
                        loss += mask_loss
                        print_losses.append(mask_loss.item() * nTotal)
                        n_totals += nTotal
                else:
                    for t in range(max_target_len):
                        decoder_output, decoder_hidden = self.decoder(
                            decoder_input, decoder_hidden, encoder_outputs
                        )
                        # No teacher forcing: next input is decoder's own current output
                        _, topi = decoder_output.topk(1)
                        decoder_input = torch.LongTensor([[topi[i][0] for i in range(self.batch_size)]])
                        decoder_input = decoder_input.cuda()
                        # Calculate and accumulate loss
            
                        mask_loss, nTotal = maskNLLLoss(decoder_output, target_sent[t], target_mask[t])
                        loss += mask_loss
                        print_losses.append(mask_loss.item() * nTotal)
                        n_totals += nTotal

                # Perform backpropatation
                loss.backward()
                count += 1
                # Clip gradients: gradients are modified in place
                _ = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip)
                _ = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip)

                # Adjust model weights
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()
                pbar(count, {'loss': sum(print_losses) / n_totals})
            torch.save(self.encoder, "encoder.pth")
            torch.save(self.decoder, "decoder.pth")
            torch.save(self.embedding,"embedding.pth")

    def evaluate(self):
        embedding = torch.load("embedding.pth")
        encoder = EncoderRNN(self.hidden_dim, embedding, self.encoder_n_layers, self.dropout).cuda()
        decoder = LuongAttnDecoderRNN(self.attn_model, embedding, self.hidden_dim, self.vocabSize, self.decoder_n_layers, self.dropout).cuda()
        
        encoder = torch.load("encoder.pth")
        decoder = torch.load("decoder.pth")

        encoder = encoder.cuda()
        decoder = decoder.cuda()

        from data import load_test_set
        source_id, source_mask, source_length = load_test_set()
        source_id = source_id.cuda()
        source_mask = source_mask.cuda()
        source_length = source_length.cuda()
        searcher = GreedySearchDecoder(encoder, decoder)
        print(source_length.shape)
        tokens, scores = searcher(source_id, source_length, 20)
        from data import convertid2sentence
        sentence = convertid2sentence(tokens)
        print(" ".join(sentence))


@ex.main
def main(_config, _run):
    print("----Runing----")
    Train = True
    trainer = Trainer(_config, _run) 
    trainer.train()
    #trainer.evaluate()





if __name__ == "__main__":
    r = ex.run()
    print(r.config)
    print(r.host_info)