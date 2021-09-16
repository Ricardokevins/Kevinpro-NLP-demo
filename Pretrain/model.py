import transformer
from transformers import BertModel
import math
max_length=128

import torch
import torch.nn as nn
from torch.nn import functional as F
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

setup_seed(44)

class BiRNN(nn.Module):
    def __init__(self, vocab=30000, embed_size=512, num_hiddens=512, num_layers=2):
        super(BiRNN, self).__init__()
        self.embedding = nn.Embedding(vocab, embed_size)
        
        self.encoder = nn.LSTM(input_size=embed_size, 
                                hidden_size=num_hiddens, 
                                num_layers=num_layers,
                                bidirectional=True, dropout=0.5)
    
        self.decoder = nn.Linear(2*num_hiddens, vocab)

    def forward(self, inputs):
        embeddings = self.embedding(inputs.permute(1, 0))
        outputs, _ = self.encoder(embeddings) # output, (h, c)
        result = self.decoder(outputs)
        return result,None

class BertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BertModel.from_pretrained('./bert-base-chinese')
        self.dropout = nn.Dropout(0.1,inplace=False)
        self.fc = nn.Linear(768, 2)
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, x):
        pad_id=0
        mask = ~(x == pad_id)
        x = self.encoder(x, attention_mask=mask)[0]
        x = x[:, 0, :]
        x = self.dropout(x)
        x = self.fc(x)    
        return x,None

class TransformerClasssifier(nn.Module):
    def __init__(self,vocab):
        super(TransformerClasssifier, self).__init__()
        self.hidden_size=512
        BaseModel = transformer.Transformer(n_src_vocab=vocab,max_length=max_length, n_layers=6, n_head=8, d_word_vec=512, d_model=512, d_inner_hid=768, dropout=0.1, dim_per_head=None)
        self.encoder = BaseModel.get_model()
        self.fc = nn.Linear(self.hidden_size, vocab)
    
    def save_pretrained_model(self):
        torch.save(self.encoder, 'pretrained.pth') 

    def forward(self, input_ids):
        sequence_heatmap,sent = self.encoder(input_ids)
        return self.fc(sequence_heatmap),None
    
