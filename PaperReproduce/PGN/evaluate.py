from __future__ import unicode_literals, print_function, division
import os
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

import config
from dataloader import Tokenizer, SumDataset
from config import SENTENCE_START, SENTENCE_END, OOV, PAD

warnings.filterwarnings('ignore')
use_cuda = config.use_gpu and torch.cuda.is_available()


class Loss_Eval:
    def __init__(self, model_path):
        self.tokenizer = Tokenizer()
        self.dataset = SumDataset(mode='Dev')
        self.dataloader = DataLoader(self.dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)
        self.model = Model(model_path, is_eval=True)

    def eval_one_batch(self, enc_batch, dec_batch):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            enc_batch
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            dec_batch

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_1 = self.model.reduce_state(encoder_hidden)

        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                                                           encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                                                           extra_zeros, enc_batch_extend_vocab, coverage, di)
            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)
            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage

            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_step_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_step_losses / dec_lens_var
        loss = torch.mean(batch_avg_loss)
        return loss.data

    def eval(self):
        total_loss = 0
        pbar = ProgressBar(n_total=len(self.dataloader), desc='Training')
        iter = 0
        for enc_input, enc_input_ext, dec_input, dec_output, enc_len, dec_len, oov_word_num in self.dataloader:
            # computing Mask of input and output
            enc_padding_mask = torch.zeros((config.batch_size, config.max_enc_steps))
            for i in range(len(enc_len)):
                for j in range(enc_len[i]):
                    enc_padding_mask[i][j] = 1

            dec_padding_mask = torch.zeros((config.batch_size, config.max_dec_steps))
            for i in range(len(dec_len)):
                for j in range(dec_len[i]):
                    dec_padding_mask[i][j] = 1

            max_oov_num = max(oov_word_num).numpy()

            # Packup input and output data to Match the origin API
            enc_batch = enc_input
            extra_zeros = None
            if max_oov_num > 0:
                extra_zeros = torch.zeros((config.batch_size, max_oov_num))

            c_t_1 = torch.zeros((config.batch_size, 2 * config.hidden_dim))
            coverage = torch.zeros(enc_batch.size())

            dec_batch = dec_input
            target_batch = dec_output
            max_dec_len = max(dec_len).numpy()

            if use_cuda:
                enc_batch = enc_batch.cuda()
                enc_padding_mask = enc_padding_mask.cuda()
                enc_len = enc_len.int()
                enc_input_ext = enc_input_ext.cuda()
                if extra_zeros is not None:
                    extra_zeros = extra_zeros.cuda()
                c_t_1 = c_t_1.cuda()
                coverage = coverage.cuda()

                dec_batch = dec_batch.cuda()
                dec_padding_mask = dec_padding_mask.cuda()
                #max_dec_len = max_dec_len.cuda()
                dec_len = dec_len.cuda()
                target_batch = target_batch.cuda()

            # Pack data for training
            enc_batch_pack = (enc_batch, enc_padding_mask, enc_len, enc_input_ext, extra_zeros, c_t_1, coverage)
            dec_batch_pack = (dec_batch, dec_padding_mask, max_dec_len, dec_len, target_batch)

            # Training
            loss = self.eval_one_batch(enc_batch_pack, dec_batch_pack)
            total_loss += loss
            iter += 1
            pbar(iter, {'loss': total_loss/iter})


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


def Eval():
    evaluator = Loss_Eval("./save/model_0_1631411954")
    #evaluator = Decode_Eval("./save/model_0_1631411954")
    evaluator.eval()


Eval()
