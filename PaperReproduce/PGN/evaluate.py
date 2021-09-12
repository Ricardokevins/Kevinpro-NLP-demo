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


class Decode_Eval:
    def __init__(self, model_path):
        self.tokenizer = Tokenizer()
        self.dataset = SumDataset(mode='Dev')
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False, drop_last=True)
        self.model = Model(model_path, is_eval=True)

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    def decode_one_batch(self, enc_batch, dec_batch):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_0, coverage_t_0 = \
            enc_batch
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            dec_batch
        
        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_0 = self.model.reduce_state(encoder_hidden)

        dec_h, dec_c = s_t_0  # 1 x 2*hidden_size
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        # decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[self.tokenizer.word2id[SENTENCE_START]],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
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

            coverage_t_1 = None
            if config.is_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)

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

                for j in xrange(config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                        log_prob=topk_log_probs[i, j].item(),
                                        state=state_i,
                                        context=context_i,
                                        coverage=coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab.word2id(data.STOP_DECODING):
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
            best_summary = self.decode_one_batch(enc_batch_pack, dec_batch_pack)
            output_ids = [int(t) for t in best_summary.tokens[1:]]
            print(output_ids)
            exit()
            # decoded_words = data.outputids2words(output_ids, self.vocab,
            #                                      (batch.art_oovs[0] if config.pointer_gen else None))
            iter += 1
            pbar(iter, {})


def Eval():
    #evaluator = Loss_Eval("./save/model_0_1631411954")
    evaluator = Decode_Eval("./save/model_0_1631411954")
    evaluator.eval()


Eval()
