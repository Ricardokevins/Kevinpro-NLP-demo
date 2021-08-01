from dataloader import SumDataset
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import FileStorageObserver
from torch import optim
from model import Encoder
from model import Decoder
from model import ReduceState
from model import Attention
from EasyTransformer.util import ProgressBar
from torch.nn.utils import clip_grad_norm_
from transformers import AdamW
from dataloader import load_test_set
import os
import time
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
ex = Experiment('PGNSum', save_git_info=False)
#ex.captured_out_filter = apply_backspaces_and_linefeeds
#obv = MongoObserver(url="localhost", port=27017, db_name="PgnSum")

ex.observers.append(FileStorageObserver('Run'))
#ex.observers.append(obv)


@ex.config
def config():
    batch_size = 64
    epochs = 20
    cuda_device = [0]
    optimizer = {'choose': 'AdamW', 'lr': 2e-3, 'eps': 1e-8}
    #optimizer = {'choose': 'SGD', 'lr': 0.01}
    loss_func = 'CE'
    possessbar = False
    vocabSize = 50000
    emb_dim = 64
    hidden_dim = 128

import numpy as np

class beam_state:
    def __init__(self, s_t_1, c_t_1, coverage, tokens):
        self.s_t_1, self.c_t_1, self.coverage = s_t_1, c_t_1, coverage
        self.score = [0]
        self.tokens = tokens
    @property
    def avg_log_prob(self):
        #print("count",self.score)
        return sum(self.score) / len(self.tokens)

def sort_beams(beams):
    return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

class Trainer:
    def __init__(self, config, runner):
        self.runner = runner
        self.use_possessbar = config['possessbar']
        self.batch_size = config["batch_size"]
        self.epoch = config['epochs']
        self.device_ids = config["cuda_device"]

        self.vocabSize = config["vocabSize"]
        self.emb_dim = config["emb_dim"]
        self.hidden_dim = config["hidden_dim"]

        print("----- get model ----")
        self.get_model()
        params = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.reduce_state.parameters())
        if config['optimizer']['choose'] == "AdamW":
            self.optim = AdamW(params, lr=config['optimizer']['lr'], eps=config['optimizer']['eps'])
        elif config['optimizer']['choose'] == "SGD":
            self.optim = optim.SGD(params, lr=config['optimizer']['lr'])
        else:
            print("Hit error")
            exit()

        if config['loss_func'] == "CE":
            self.criterion = nn.CrossEntropyLoss()

        data = SumDataset()
        self.dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=True)

        # self.model=nn.DataParallel(self.model,device_ids=self.device_ids)

    def get_model(self):
        # self.encoder = Encoder(self.vocabSize, self.emb_dim, self.hidden_dim)
        # self.decoder = Decoder(self.vocabSize, self.emb_dim, self.hidden_dim)
        # self.reduce_state = ReduceState(self.hidden_dim)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.reduce_state = ReduceState()
        self.encoder = self.encoder.cuda()
        self.decoder = self.decoder.cuda()
        self.reduce_state = self.reduce_state.cuda()

    def train(self):
        print("---- Start Training ----")
        log_freq = int(len(self.dataloader) / 10)

        pbar = ProgressBar(n_total=len(self.dataloader), desc='Training')

        best_loss = 1000000
        # self.encoder = torch.load("./model/encoder.pth")
        # self.decoder = torch.load("./model/decoder.pth")
        # self.reduce_state = torch.load("./model/reduce_state.pth")
        
        for epo in range(self.epoch):
            count = 0
            total = 0
            loss_sum = 0
            avg_loss = 0

            acc = 0
            total_acc = 0
            self.enc_lens = np.zeros((self.batch_size), dtype=np.int32)
            self.encoder.train()
            self.decoder.train()
            self.reduce_state.train()
            for source, target, source_ext, target_ext, oov_num,source_length,target_length in self.dataloader:
                self.optim.zero_grad()
                if source.size(0) != self.batch_size:
                    break
                source, target, source_ext, target_ext, oov_num = source.cuda(), target.cuda(), source_ext.cuda(), target_ext.cuda(), oov_num.cuda()
                
                source_ext = source_ext[:, : torch.max(source_length).item()]
                #print(source_ext.shape)
                source_mask = torch.zeros(self.batch_size, torch.max(source_length).item()).cuda()
                for i in range(self.batch_size):
                    for j in range(source_length[i]):
                        source_mask[i][j] = 1

                target_mask = torch.zeros(self.batch_size, torch.max(target_length).item()).cuda()
                for i in range(self.batch_size):
                    for j in range(target_length[i]):
                        target_mask[i][j] = 1
  
                # source_mask = (source_ext != 0).float().cuda()
                # target_mask = (target_ext != 0).float().cuda()
                #print(source_mask)
                #print(source_ext)

                coverage = torch.zeros(self.batch_size, torch.max(source_length).item()).cuda()
                c_t_1 = torch.zeros((self.batch_size, 2 * self.hidden_dim)).cuda()
                encoder_outputs, encoder_feature, encoder_hidden = self.encoder(source,source_length)
                s_t_1 = self.reduce_state(encoder_hidden)
                step_losses = []

                decode_result = []
                target_result = []
                for di in range(torch.max(target_length).item()):
                    y_t_1 = target[:, di]  # 摘要的一个单词，batch里的每个句子的同一位置的单词编码
                    #print(y_t_1)
                    max_art_oovs = torch.max(oov_num)
                    extra_zeros = torch.zeros((self.batch_size, max_art_oovs)).cuda()
                    final_dist, s_t_1,  c_t_1, attn_dist, p_gen, next_coverage = self.decoder(y_t_1, s_t_1,
                                                                                              encoder_outputs, encoder_feature, source_mask, c_t_1,
                                                                                              extra_zeros, source_ext, coverage, di)
                    target_word = target_ext[:, di+1]  # 摘要的下一个单词的编码
                    
                    num, index = torch.max(final_dist, 1)
                    acc += (target_word == index).sum()
                    #decode_result.append(index.item())
                    #target_result.append(target_word.item())
                    total_acc += self.batch_size
                    #print('target',target_word)
                    #print('predict',index)
                    gold_probs = torch.gather(final_dist, 1, target_word.unsqueeze(1)).squeeze()   # 取出目标单词的概率gold_probs
                    step_loss = -torch.log(gold_probs+ 1e-12)  # 最大化gold_probs，也就是最小化step_loss（添加负号）

                    step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                    step_loss = step_loss + step_coverage_loss
                    coverage = next_coverage

                    step_mask = target_mask[:, di]
                    step_loss = step_loss * step_mask
                    step_losses.append(step_loss)
    
                sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
                #batch_avg_loss = sum_losses/dec_lens_var
                #loss = torch.mean(batch_avg_loss)
                avg_len = torch.sum(target_mask) / self.batch_size
                sum_losses = sum_losses/avg_len
                loss = torch.mean(sum_losses)
                loss.backward()

                total += self.batch_size
                count += 1
                avg_loss += loss.item()
                if count % log_freq == 0:
                    self.runner.log_scalar("Loss", avg_loss / count)
                    self.runner.log_scalar("Acc", acc.item()/total_acc)
                pbar(count, {'loss': avg_loss/count})
                clip_grad_norm_(self.encoder.parameters(), 2.0)
                clip_grad_norm_(self.decoder.parameters(), 2.0)
                clip_grad_norm_(self.reduce_state.parameters(), 2.0)
                self.optim.step()
                torch.cuda.empty_cache()

            if avg_loss < best_loss:
                print("---- save with best loss ----")
                best_loss = avg_loss
                file_path = './model'
                # os.mkdir(file_path)
                torch.save(self.encoder, file_path+'/'+"encoder.pth")
                torch.save(self.decoder, file_path+'/'+"decoder.pth")
                torch.save(self.reduce_state, file_path+'/'+"reduce_state.pth")
            #self.greedy_decoder()

    def greedy_decoder(self):

        self.encoder = torch.load("./model/encoder.pth")
        self.decoder = torch.load("./model/decoder.pth")
        self.reduce_state = torch.load("./model/reduce_state.pth")

        self.encoder.eval()
        self.decoder.eval()
        self.reduce_state.eval()

        source, source_ext, oov_num,source_length, sent, oov_dict, tokenizer = load_test_set()
        decode_words = []
        decode_ids = 2
        with torch.no_grad():
            source, source_ext = source.cuda(),  source_ext.cuda()
            source_ext = source_ext[:, : torch.max(source_length).item()]
            #print(source_ext.shape)
            coverage = torch.zeros(torch.max(source_length).item()).cuda()
            source_mask = torch.zeros(torch.max(source_length).item()).cuda()
            
            for j in range(source_length):
                source_mask[j] = 1
            c_t_1 = torch.zeros((1, 2 * self.hidden_dim)).cuda()
            encoder_outputs, encoder_feature, encoder_hidden = self.encoder(source,source_length)
            s_t_1 = self.reduce_state(encoder_hidden)
            step_losses = []
            for di in range(10):
                y_t_1 = torch.tensor([decode_ids]).cuda()     # 摘要的一个单词，batch里的每个句子的同一位置的单词编码
                max_art_oovs = torch.max(oov_num)
                extra_zeros = torch.zeros((1, max_art_oovs)).cuda()
                final_dist, s_t_1,  c_t_1, attn_dist, p_gen, next_coverage = self.decoder(y_t_1, s_t_1,
                                                                                          encoder_outputs, encoder_feature, source_mask, c_t_1,
                                                                                          extra_zeros, source_ext, coverage, di)

                num, index = torch.max(final_dist, 1)

                index = index.item()
                if index < 50000:
                    word = tokenizer.idx2word[index]

                    decode_words.append(word)
                    decode_ids = index
                    if word == SENTENCE_END:
                        break
                else:
                    decode_words.append(oov_dict[index-50000])
                    decode_ids = tokenizer.word2idx['<OOV>']
 
                coverage = next_coverage
        print(decode_words[:10])

    def beam_decoder(self):
        self.encoder = torch.load("./model/encoder.pth")
        self.decoder = torch.load("./model/decoder.pth")
        self.reduce_state = torch.load("./model/reduce_state.pth")

        self.encoder.eval()
        self.decoder.eval()
        self.reduce_state.eval()

        source, source_ext, oov_num,source_length, sent, oov_dict, tokenizer = load_test_set()
        results = []
        decode_words = []
        decode_ids = 2
        with torch.no_grad():
            source_ext = source_ext[:, : torch.max(source_length).item()]
            source, source_ext = source.cuda(),  source_ext.cuda()
            source_mask = torch.zeros(5, torch.max(source_length).item()).cuda()
            for i in range(5):
                for j in range(source_length[i]):
                    source_mask[i][j] = 1    
            coverage = torch.zeros(5, torch.max(source_length).item()).cuda()          
            c_t_1 = torch.zeros((1, 2 * self.hidden_dim)).cuda()
            encoder_outputs, encoder_feature, encoder_hidden = self.encoder(source,source_length)
            s_t_1 = self.reduce_state(encoder_hidden)

            dec_h, dec_c = s_t_1 # 1 x 2*hidden_size
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            y_t_1 = torch.tensor([decode_ids]).cuda()

            beam_list = [beam_state((dec_h[0], dec_c[0]), c_t_1[0], coverage,[2]) for i in range(5)]

            for di in range(100):
                last_token = [b.tokens[-1] for b in beam_list]
                last_token = [t if t < 50000 else tokenizer.word2idx['<OOV>'] for t in last_token]
                y_t_1 = torch.LongTensor(last_token).cuda()
    
                
                all_state_h =[]
                all_state_c = []
                all_context = []

                for h in beam_list:
                    state_h, state_c = h.s_t_1
                    all_state_h.append(state_h)
                    all_state_c.append(state_c)
                    all_context.append(h.c_t_1)
                

                all_coverage = []
                for h in beam_list:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)

                num = len(all_context)
                s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
                c_t_1 = torch.stack(all_context, 0)

                max_art_oovs = torch.max(oov_num)
                extra_zeros = torch.zeros((num, max_art_oovs)).cuda()
                final_dist, s_t_1,  c_t_1, attn_dist, p_gen, next_coverage = self.decoder(y_t_1, s_t_1,
                                                                                            encoder_outputs, encoder_feature, source_mask, c_t_1,
                                                                                            extra_zeros, source_ext, coverage, di)

                log_probs = torch.log(final_dist)
                topk_log_probs, topk_ids = torch.topk(log_probs, 5)

                all_beams = []
                num_orig_beams = 1 if di == 0 else len(beam_list)
                for i in range(num_orig_beams):
                    h = beam_list[i]
                    state_i = (dec_h[i], dec_c[i])
                    context_i = c_t_1[i]
                    coverage_i = (next_coverage[i])
                    
                    for j in range(5):  # for each of the top 2*beam_size hyps:
                        new_score = h.score.copy()
                        new_score.append(topk_log_probs[i, j].item())
                        new_tokens = h.tokens.copy()
                        new_tokens.append(topk_ids[i, j].item())
                        new_beam = beam_state(state_i,context_i,coverage_i,new_tokens)
                        #print(new_score)
                        new_beam.score = new_score

                        all_beams.append(new_beam)

                beam_list = []


                for h in sort_beams(all_beams):
                    if h.tokens[-1] == tokenizer.word2idx[SENTENCE_END]:
                        results.append(h)
                    else:
                        beam_list.append(h)
                    if len(beam_list) == 5 or len(results) == 5:
                        break

        if len(results) == 0:
            results = beam_list
        beams_sorted = sort_beams(results)
        best = beams_sorted[0]
        for index in best.tokens:
            if index < 50000:
                word = tokenizer.idx2word[index]
                decode_words.append(word)
                decode_ids = index
                if word == SENTENCE_END:
                    break
            else:
                decode_words.append(oov_dict[index-50000])
                decode_ids = tokenizer.word2idx['<OOV>']

        print(decode_words[:10])

@ex.main
def main(_config, _run):
    print("----Runing----")
    Train = False
    trainer = Trainer(_config, _run)
    if Train:
        trainer.train()
    else:
        #trainer.greedy_decoder()
        trainer.beam_decoder()


if __name__ == "__main__":
    print("Hello")
    r = ex.run()
    print(r.config)
    print(r.host_info)
