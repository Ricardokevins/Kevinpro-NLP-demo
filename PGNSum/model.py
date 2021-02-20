import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self,vocabSize,emb_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocabSize, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.W_h = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
    
    def forward(self, input):
        embedded = self.embedding(input_x)
        output, hidden = self.lstm(packed)  # hidden is tuple([2, batch, hid_dim], [2, batch, hid_dim])
        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)  # [batch, max(seq_lens), 2*hid_dim]
        encoder_outputs = encoder_outputs.contiguous()                      # [batch, max(seq_lens), 2*hid_dim]
        encoder_feature = encoder_outputs.view(-1, 2*config.hidden_dim)  
        encoder_feature = self.W_h(encoder_feature)       # [batch*max(seq_lens), 2*hid_dim]
        return encoder_outputs, encoder_feature, hidden   # [B, max(seq_lens), 2*hid_dim], [B*max(seq_lens), 2*hid_dim], tuple([2, batch, hid_dim], [2, batch, hid_dim])


class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()

        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_h)
        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_c)

    def forward(self, hidden):
        h, c = hidden    # h, c dim = [2, batch, hidden_dim]
        h_in = h.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)  # [batch, hidden_dim*2]
        hidden_reduced_h = F.relu(self.reduce_h(h_in))                         # [batch, hidden_dim]
        c_in = c.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))

        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0))  # h, c dim = [1, batch, hidden_dim]


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        if config.is_coverage:
            self.W_c = nn.Linear(1, config.hidden_dim * 2, bias=False)
        self.decode_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)

    def forward(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage):
        b, t_k, n = list(encoder_outputs.size())

        dec_fea = self.decode_proj(s_t_hat)               # B x 2*hid_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous() # B x t_k x 2*hid_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)   # B * t_k x 2*hid_dim

        att_features = encoder_feature + dec_fea_expanded # B * t_k x 2*hidden_dim
        if config.is_coverage:
            coverage_input = coverage.view(-1, 1)         # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)   # B * t_k x 2*hidden_dim
            att_features = att_features + coverage_feature

        e = torch.tanh(att_features)   # B * t_k x 2*hidden_dim
        scores = self.v(e)             # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k

        attn_dist_ = F.softmax(scores, dim=1)*enc_padding_mask # B x t_k
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)           # B x 1 x t_k
        c_t = torch.bmm(attn_dist, encoder_outputs)  # B x 1 x n
        c_t = c_t.view(-1, config.hidden_dim * 2)    # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, t_k)          # B x t_k

        if config.is_coverage:
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage





class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.attention_network = Attention()
        # decoder
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
    

        self.x_context = nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.emb_dim)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
  

        if config.pointer_gen:
            self.p_gen_linear = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)

        #p_vocab
        self.out1 = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        self.out2 = nn.Linear(config.hidden_dim, config.vocab_size)


    def forward(self, y_t_1, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask,
                c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, step):

        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1
            s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                                 c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
            # attention
            c_t, _, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                           enc_padding_mask, coverage)
            coverage = coverage_next

        y_t_1_embd = self.embedding(y_t_1)
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
  
        
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                             c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
        c_t, attn_dist, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                               enc_padding_mask, coverage)

        if self.training or step > 0:
            coverage = coverage_next

        p_gen = None
        if config.pointer_gen:
            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = torch.sigmoid(p_gen)

        output = torch.cat((lstm_out.view(-1, config.hidden_dim), c_t), 1) # B x hidden_dim * 3
        output = self.out1(output) # B x hidden_dim

        #output = F.relu(output)

        output = self.out2(output) # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)

        if config.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)
            
            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)

        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage