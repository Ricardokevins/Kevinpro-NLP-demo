from __future__ import unicode_literals, print_function, division
import time
import argparse
import torch
from model import Model
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adagrad
from torch.utils.data import Dataset, DataLoader
import numpy as np
from EasyTransformer.util import ProgressBar
import warnings
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from config import max_enc_steps
from config import max_dec_steps
from config import train_data_path, dev_data_path, vocab_path, vocab_size
from config import SENTENCE_START, SENTENCE_END, OOV, PAD

import config
from dataloader import Tokenizer
warnings.filterwarnings('ignore')
use_cuda = config.use_gpu and torch.cuda.is_available()

class DecodeDataset(Dataset):
    def __init__(self, input_lines):
        self.input_lines = input_lines
        self.tokenizer = Tokenizer()
        self.enc_input_list = []
        self.enc_input_ext_list = []
        self.dec_input_list = []
        self.dec_output_list = []
        self.enc_len_list = []
        self.dec_len_list = []
        self.oov_word_num_list = []
        self.oov_word_list = []
        print("dataNumber", len(input_lines))
        from tqdm import tqdm
        for line in self.input_lines:
            enc_input, enc_input_ext, enc_len, oov_word = self.get_one_sample(line)
            self.enc_input_list.append(enc_input)
            self.enc_input_ext_list.append(enc_input_ext)
            self.enc_len_list.append(enc_len)
            self.oov_word_num_list.append(len(oov_word))
            self.oov_word_list.append(oov_word)
            # print(self.enc_input_list)

        # For test to see
        # print(self.enc_input_list[0])
        # print(self.enc_input_ext_list[0])
        # print(self.dec_input_list[0])
        # print(self.dec_output_list[0])
        # print(self.enc_len_list[0])
        # print(self.dec_len_list[0])
        # print(self.oov_word_num_list[0])

    def __len__(self):
        return len(self.enc_input_list)

    def __getitem__(self, idx):
        return torch.tensor(self.enc_input_list[idx]), torch.tensor(self.enc_input_ext_list[idx]), torch.tensor(self.enc_len_list[idx]), torch.tensor(self.oov_word_num_list[idx]), self.oov_word_list[idx]

    def sentence2ids(self, article):
        ids = []
        for i in article:
            if i in self.tokenizer.word2id:
                ids.append(self.tokenizer.word2id[i])
            else:
                ids.append(self.tokenizer.word2id[OOV])
        ids = ids[:max_enc_steps]
        return ids

    def sentence2idsExt(self, article):
        ids = []
        oov_word_list = []
        for i in article:
            if i in self.tokenizer.word2id:
                ids.append(self.tokenizer.word2id[i])
            else:
                if i not in oov_word_list:
                    oov_word_list.append(i)
                oov_index = oov_word_list.index(i)
                ids.append(vocab_size+oov_index)
        ids = ids[:max_enc_steps]
        return ids, oov_word_list

    def padding_input(self, input_seq):
        while(len(input_seq) < max_enc_steps):
            input_seq.append(self.tokenizer.word2id[PAD])
        return input_seq

    def get_one_sample(self, source):
        enc_input = self.sentence2ids(source)
        enc_input_ext, oov_word = self.sentence2idsExt(source)
        enc_len = len(enc_input)
        enc_input = self.padding_input(enc_input)
        enc_input_ext = self.padding_input(enc_input_ext)

        assert max_enc_steps == len(enc_input)
        assert max_enc_steps == len(enc_input_ext)

        return enc_input, enc_input_ext, enc_len, oov_word

class Beam(object):
    def __init__(self, tokens, log_probs, state, context, coverage):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.context = context
        self.coverage = coverage

    def extend(self, token, log_prob, state, context, coverage):
        return Beam(tokens=self.tokens + [token],
                    log_probs=self.log_probs + [log_prob],
                    state=state,
                    context=context,
                    coverage=coverage)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)

class Decode_Eval:
    def __init__(self, model_path, input_lines):
        self.tokenizer = Tokenizer()
        self.dataset = DecodeDataset(input_lines)
        self.model = Model(model_path, is_eval=True)

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    def decode_one_batch(self, enc_batch):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_0, coverage_t_0 = \
            enc_batch
        

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_0 = self.model.reduce_state(encoder_hidden)

        dec_h, dec_c = s_t_0  # 1 x 2*hidden_size
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        encoder_outputs = [encoder_outputs for i in range(config.beam_size)]
        encoder_feature = [encoder_feature for i in range(config.beam_size)]
        #extra_zeros = [extra_zeros for i in config.beam_size]

        encoder_outputs = torch.stack(encoder_outputs,1).squeeze()
        #print('encpder',encoder_outputs.shape)
        encoder_feature = torch.stack(encoder_feature,1).squeeze().reshape(-1,2*config.hidden_dim)

        # decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[self.tokenizer.word2id[SENTENCE_START]],
                      log_probs=[0.0],
                      state=(dec_h, dec_c),
                      context=c_t_0[0],
                      coverage=(coverage_t_0[0] if config.is_coverage else None))
                 for _ in range(config.beam_size)]
        results = []
        steps = 0
        while steps < config.max_dec_steps and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < len(self.tokenizer.word2id) else self.tokenizer.word2id[OOV]
                             for t in latest_tokens]
            y_t_1 = torch.LongTensor(latest_tokens)
            if use_cuda:
                y_t_1 = y_t_1.cuda()

            all_state_h = []
            all_state_c = []
            all_context = []

            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)
                all_context.append(h.context)

            s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            c_t_1 = torch.stack(all_context, 0)

            #print("test",s_t_1[0].shape,s_t_1[1].shape)
            #exit()
            coverage_t_1 = None
            if config.is_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)

            # print(coverage_t_1.shape)
            # print(c_t_1.shape)
            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.model.decoder(y_t_1, s_t_1,
                                                                                    encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                                                    extra_zeros, enc_batch_extend_vocab, coverage_t_1, steps)
            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, config.beam_size * 2)

            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]
                coverage_i = (coverage_t[i] if config.is_coverage else None)

                for j in range(config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                        log_prob=topk_log_probs[i, j].item(),
                                        state=state_i,
                                        context=context_i,
                                        coverage=coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.tokenizer.word2id[SENTENCE_END]:
                    if steps >= config.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == config.beam_size or len(results) == config.beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)

        return beams_sorted[0]

    def eval(self):
        pbar = ProgressBar(n_total=len(self.dataset), desc='Training')
        iter = 0
        f = open(config.output_path,'w',encoding='utf-8')
        for enc_input, enc_input_ext, enc_len, oov_word_num, oov_word_list in self.dataset:
            enc_input = enc_input.reshape(1,-1)
            enc_input_ext = enc_input.reshape(1,-1)
            enc_len = enc_len.reshape(1,-1)
            oov_word_num = oov_word_num.reshape(1,-1)

            # enc_input_list = [enc_input for i in range(config.beam_size)]
            # enc_input_ext_list = [enc_input_ext for i in range(config.beam_size)]
            # enc_len_list = [enc_len for i in range(config.beam_size)]
            # enc_input = torch.stack(enc_input_list, 0)
            # enc_input_ext = torch.stack(enc_input_ext_list, 0)
            # enc_len = torch.stack(enc_len_list, 0)
            
            # computing Mask of input and output
            enc_padding_mask = torch.zeros(enc_input.shape)
            for i in range(len(enc_len)):
                for j in range(enc_len[i]):
                    enc_padding_mask[i][j] = 1
            max_oov_num = max(oov_word_num).numpy().tolist()[0]
            
            #'''test
            import types
            if isinstance(max_oov_num, int) is False:
                print(max_oov_num)
            #'''
            # Packup input and output data to Match the origin API
            enc_batch = enc_input
            extra_zeros = None
            if max_oov_num > 0:
                extra_zeros = torch.zeros((config.beam_size, max_oov_num))

            c_t_1 = torch.zeros((1, 2 * config.hidden_dim))
            coverage = torch.zeros(1,enc_batch.shape[1])

            

            if use_cuda:
                enc_batch = enc_batch.cuda()
                enc_padding_mask = enc_padding_mask.cuda()
                enc_len = enc_len.int()
                enc_input_ext = enc_input_ext.cuda()
                if extra_zeros is not None:
                    extra_zeros = extra_zeros.cuda()
                c_t_1 = c_t_1.cuda()
                coverage = coverage.cuda()

                

            # Pack data for training
            enc_batch_pack = (enc_batch, enc_padding_mask, enc_len, enc_input_ext, extra_zeros, c_t_1, coverage)


            # Training
            best_summary = self.decode_one_batch(enc_batch_pack)
            output_ids = [int(t) for t in best_summary.tokens[1:]]
            
            # decoded_words = data.outputids2words(output_ids, self.vocab,
            #                                      (batch.art_oovs[0] if config.pointer_gen else None))
            words = self.convertId2words(output_ids,oov_word_list)
            summary = "".join(words)
            f.write(summary + '\n')
            iter += 1
            pbar(iter, {})

    def convertId2words(self,output_ids,oov_dict):
        words = []
        for i in output_ids:
            if i >= config.vocab_size:
                words.append(oov_dict[i-config.vocab_size])
            else:
                words.append(self.tokenizer.id2word[i])
        return words


def decode():
    load_f = open(dev_data_path, 'r', encoding='utf-8')
    temp = load_f.readlines()
    print("dataNumber", len(temp))
    input_lines = []
    from tqdm import tqdm
    for line in tqdm(temp):
        dic = json.loads(line)
        source = dic['content']
        input_lines.append(source)

    model_path = "./save/model_0_1631416384"
    de = Decode_Eval(model_path, input_lines)
    de.eval()


decode()
