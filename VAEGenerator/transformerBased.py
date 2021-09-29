from transformers import BertTokenizer
from transformers import BertModel
import torch
from tqdm import tqdm
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from EasyTransformer.util import ProgressBar
from transformers import AdamW
from model import TransformerVAE
import torch.optim as optim

pretrain_path = '../../../PTM//bert-base-chinese'
bert_tokenizer = BertTokenizer.from_pretrained(pretrain_path)
corpus_path = './toutiaonews38w/train.tsv'

f = open(corpus_path, 'r',encoding='utf8')
lines = f.readlines()

# print(lines[1].strip().split('\t'))

count = 0
max_length = 10
batch_size = 8

SOSID = bert_tokenizer.encode('[CLS]', add_special_tokens=False)[0]
EOSID = bert_tokenizer.encode('[SEP]', add_special_tokens=False)[0]
# print(SOSID,EOSID)
# exit()
# test add token
# sent = '[SOS]我喜欢这个。[EOS]'
# indexed_tokens = bert_tokenizer.encode(sent, add_special_tokens=False)
# print(indexed_tokens)
# tokens = bert_tokenizer.convert_ids_to_tokens(indexed_tokens)
# print(tokens)
# exit()
enc_inputs = []
dec_inputs = []
dec_targets = []

def padding(idx):
    padded_idx = idx.copy()
    while len(padded_idx)<max_length:
        padded_idx.append(0)
    return padded_idx

for i in tqdm(range(1,len(lines))):
    sent = lines[i].strip().split('\t')[1]
    if len(sent) < max_length-1:
        count += 1
        #indexed_tokens = bert_tokenizer.encode(sent, add_special_tokens=True)
        indexed_tokens = bert_tokenizer.encode(sent, add_special_tokens=False)
        enc_input = padding(indexed_tokens)
        dec_input = padding([SOSID]+indexed_tokens)
        dec_target = padding(indexed_tokens+[EOSID])
        enc_inputs.append(enc_input)
        dec_inputs.append(dec_input)
        dec_targets.append(dec_target)
    

print("Trimmed Sample count: ",count)
dict_size = len(bert_tokenizer.vocab)
print("Vocab Size: ",dict_size)
enc_inputs = torch.tensor(enc_inputs)
dec_inputs = torch.tensor(dec_inputs)
dec_targets = torch.tensor(dec_targets)
train_dataset = torch.utils.data.TensorDataset(enc_inputs,dec_inputs,dec_targets)
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
crossentropyloss = nn.CrossEntropyLoss()

#bert_sent.append(indexed_tokens)
#tokens = bert_tokenizer.convert_ids_to_tokens(indexed_tokens)
#print(tokens)
def loss_function(x_hat, x, mu, log_var):
    """
    Calculate the loss. Note that the loss includes two parts.
    :param x_hat:
    :param x:
    :param mu:
    :param log_var:
    :return: total loss, BCE and KLD of our model
    """
    # 1. the reconstruction loss.
    # We regard the MNIST as binary classification
    # print(x)
    # print(x_hat)
    # print(mu.shape)
    # print(log_var.shape)
    # exit()
    #batch_size = x_hat.shape[0]
    #x_hat = x_hat.reshape(batch_size,-1)
    x_hat = x_hat.reshape(-1,dict_size)
    x = x.reshape(-1)
    # print(x_hat)
    # print(x)
    CEL = crossentropyloss(x_hat, x)

    # 2. KL-divergence
    # D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
    # here we assume that \Sigma is a diagonal matrix, so as to simplify the computation
    KLD = 0.5 * torch.sum(torch.exp(log_var) + torch.pow(mu, 2) - 1. - log_var)

    # 3. total loss
    loss = CEL + KLD
    #loss = CEL
    return loss, CEL, KLD

net = TransformerVAE().cuda()
net=torch.load('model/epoch76.pt')
# net = VAE().cuda()
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.99)
#optimizer = AdamW(net.parameters(),lr = 2e-5, eps = 1e-8)
index = 0
loss_sum = 0 



# def model_test(model):
#     z = 
import random
import numpy as np
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(500)

def decode_test(model, start_symbol):
    z = (torch.randn(1, 10 ,400)*2+11.5).cuda() # 每一行是一个隐变量，总共有batch_size行
    #print(z)
    enc_input = torch.tensor([3 for i in range(10)]).unsqueeze(0).cuda()
    dec_input = torch.zeros(1, 0).type_as(enc_input.data).cuda()
    terminal = False
    next_symbol = start_symbol
    token_list = []
    for i in range(10):         
        dec_input=torch.cat([dec_input.detach(),torch.tensor([[next_symbol]],dtype=enc_input.dtype).cuda()],-1)
        #print(dec_input)
        #print(enc_input)
        dec_outputs = model.decode(dec_input, enc_input, z)
        #print("hello?")
        prob = dec_outputs.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[-1]
        next_symbol = next_word
        # if next_symbol == tgt_vocab["EOS"]:
        #     terminal = True

        token_list.append(next_word )
    print(bert_tokenizer.convert_ids_to_tokens(token_list))        
    return dec_input

# def decode_test(model):
#     z = torch.randn(1, 10 ,400).cuda() # 每一行是一个隐变量，总共有batch_size行
    
#     # 对隐变量重构
#     with torch.no_grad():
#         random_res = model.decode(z)

#     _, result = torch.max(random_res, 2)
#     #print(result.shape)
#     # exit()
#     # exit()
#     #predict = result.cpu().numpy().tolist()
#     #print(predict)
#     #for i in result:
#     #print(result)
#     result = result[0]
#     #print(result.shape)
#     tokens = bert_tokenizer.convert_ids_to_tokens(result)
#     print(tokens)
#     #exit()
    
decode_test(net,SOSID)
exit()
for epoch in range(100):
    print("\nstart Epoch: ",epoch)
    loss_sum = 0
    index = 0
    pbar = ProgressBar(n_total=len(train_iter), desc='Training')
    for enc_input,dec_input,dec_target in train_iter:
        enc_input = enc_input.cuda()
        dec_input = dec_input.cuda()
        dec_target = dec_target.cuda()
        x_hat, mu, log_var = net(enc_input,dec_input)
        loss, CEL, KLD = loss_function(x_hat,dec_target,mu,log_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        index+=1
        loss_sum +=loss
        pbar(index-1, {'loss': loss_sum/index})
        if index % 20 == 0:
            print("\n")
            #decode_test(net)
            #print(CEL,KLD,loss)
            x_hat = x_hat[0]
            x_hat = x_hat.unsqueeze(0)
            _, result = torch.max(x_hat, 2)
            result = result[0]
            #print(result.shape)
            tokens = bert_tokenizer.convert_ids_to_tokens(enc_input[0])
            print(tokens)
            tokens = bert_tokenizer.convert_ids_to_tokens(result)
            print(tokens)
    #torch.save(net, 'model/epoch{}.pt'.format(epoch)) 
    # print(loss)
    # exit()

