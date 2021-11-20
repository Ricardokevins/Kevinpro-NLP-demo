from EasyTransformer import transformer
from transformers import BertModel
import math
max_length=64
import torch
import torch.nn as nn
from torch.nn import functional as F
from config import *
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

setup_seed(44)
class BertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BertModel.from_pretrained(BERT_PATH)
        
    def forward(self, x):
        pad_id=0
        mask = ~(x == pad_id)
        x = self.encoder(x, attention_mask=mask)[0]
        x = x[:, 0, :]
        return x