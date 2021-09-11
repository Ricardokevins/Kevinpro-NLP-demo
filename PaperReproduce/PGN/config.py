import os
import torch
from numpy import random

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
OOV = '<unk>'
PAD = '<pad>'


train_data_path = './data/train.json'
dev_data_path = './data/dev.json'
vocab_path = './data/vocab.txt'
save_path = './save'

# Hyperparameters
EPOCH = 5
hidden_dim = 128
emb_dim = 64
batch_size = 16
max_enc_steps = 150
max_dec_steps = 35

beam_size = 4
# min_dec_steps = 35
min_dec_steps = 5
vocab_size = 10_000

lr = 0.15
adagrad_init_acc = 0.1
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4
max_grad_norm = 2.0

pointer_gen = True
#pointer_gen = False
is_coverage = True
#is_coverage = False
cov_loss_wt = 1.0
eps = 1e-12
eps = 1e-12
max_iterations = 750_000

lr_coverage=0.15

# 使用GPU相关
use_gpu = True
GPU = "cuda:0"
USE_CUDA = use_gpu and torch.cuda.is_available()     # 是否使用GPU
NUM_CUDA = torch.cuda.device_count()
DEVICE = torch.device(GPU if USE_CUDA else 'cpu')

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
if USE_CUDA:
    torch.cuda.manual_seed_all(SEED)
