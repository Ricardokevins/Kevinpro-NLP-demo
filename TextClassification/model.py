from EasyTransformer import transformer
from transformers import BertModel
import math
max_length=64
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
    
        self.decoder = nn.Linear(4*num_hiddens, 2)

    def forward(self, inputs):
        embeddings = self.embedding(inputs.permute(1, 0))
        outputs, _ = self.encoder(embeddings) # output, (h, c)
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        outs = self.decoder(encoding)
        return outs,None

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
        self.hidden_size=256
        BaseModel = transformer.Transformer(n_src_vocab=vocab,max_length=max_length, n_layers=3, n_head=4, d_word_vec=256, d_model=256, d_inner_hid=512, dropout=0.1, dim_per_head=None)
        self.encoder = BaseModel.get_model()
        Use_pretrain = True
        if Use_pretrain:
            print("========================= Using pretrained model =========================")
            self.encoder = torch.load('../Pretrain/pretrained.pth')
        self.fc = nn.Linear(self.hidden_size, 10)
    
    def forward(self, input_ids):
        sequence_heatmap,sent = self.encoder(input_ids)
        return self.fc(sent),None
    
class BiLSTM_Attention1(nn.Module):

    def __init__(self, vocab_size=30000, embedding_dim=512, hidden_dim=512, n_layers=2):

        super(BiLSTM_Attention1, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim * 2, 2)
        self.dropout = nn.Dropout(0.5)

    #x,query：[batch, seq_len, hidden_dim*2]
    def attention_net(self, x, query, mask=None):      #软性注意力机制（key=value=x）

        d_k = query.size(-1)    
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  #打分机制  scores:[batch, seq_len, seq_len]
        p_attn = F.softmax(scores, dim=-1)  #对最后一个维度归一化得分
        context = torch.matmul(p_attn, x).sum(1)       #对权重化的x求和，[batch, seq_len, hidden_dim*2]->[batch, hidden_dim*2]
        return context, p_attn


    def forward(self, x):
        x = x.permute(1, 0)
        embedding = self.dropout(self.embedding(x))       #[seq_len, batch, embedding_dim]

        # output: [seq_len, batch, hidden_dim*2]     hidden/cell: [n_layers*2, batch, hidden_dim]
        output, (final_hidden_state, final_cell_state) = self.rnn(embedding)
        output = output.permute(1, 0, 2)                  #[batch, seq_len, hidden_dim*2]

        query = self.dropout(output)
        attn_output, attention = self.attention_net(output, query)       #和LSTM的不同就在于这一句
        logit = self.fc(attn_output)
        return logit,attention

class BiLSTM_Attention2(nn.Module):

    def __init__(self, vocab_size=30000, embedding_dim=512, hidden_dim=512, n_layers=2):

        super(BiLSTM_Attention2, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)        #单词数，嵌入向量维度
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim * 2, 2)
        self.dropout = nn.Dropout(0.5)

        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.w_omega = nn.Parameter(torch.Tensor(hidden_dim * 2, hidden_dim * 2))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_dim * 2, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)


    def attention_net(self, x):       #x:[batch, seq_len, hidden_dim*2]

        u = torch.tanh(torch.matmul(x, self.w_omega))         #[batch, seq_len, hidden_dim*2]
        att = torch.matmul(u, self.u_omega)                   #[batch, seq_len, 1]
        att_score = F.softmax(att, dim=1)

        scored_x = x * att_score                              #[batch, seq_len, hidden_dim*2]

        context = torch.sum(scored_x, dim=1)                  #[batch, hidden_dim*2]
        return context,att_score


    def forward(self, x):
        x = x.permute(1, 0)
        embedding = self.dropout(self.embedding(x))       #[seq_len, batch, embedding_dim]

        # output: [seq_len, batch, hidden_dim*2]     hidden/cell: [n_layers*2, batch, hidden_dim]
        output, (final_hidden_state, final_cell_state) = self.rnn(embedding)
        output = output.permute(1, 0, 2)                  #[batch, seq_len, hidden_dim*2]

        attn_output,attn = self.attention_net(output)
        logit = self.fc(attn_output)
        return logit,attn