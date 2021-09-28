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
pretrain_path = '../../../PTM//bert-base-chinese'
bert_tokenizer = BertTokenizer.from_pretrained(pretrain_path)
corpus_path = './toutiaonews38w/train.tsv'

f = open(corpus_path, 'r',encoding='utf8')
datas = []
lines = f.readlines()

# print(lines[1].strip().split('\t'))

count = 0
max_length = 10
batch_size = 4

for i in tqdm(range(1,len(lines))):
    sent = lines[i].strip().split('\t')[1]
    if len(sent) < 8:
        count += 1

        indexed_tokens = bert_tokenizer.encode(sent, add_special_tokens=True)
        while len(indexed_tokens) <  max_length:
            indexed_tokens.append(0)  # 0 is id for [PAD]
        indexed_tokens = indexed_tokens[: max_length]
        datas.append(indexed_tokens)

print("Trimmed Sample count: ",count)
dict_size = len(bert_tokenizer.vocab)
print("Vocan Size: ",dict_size)
train_data = torch.tensor(datas)
train_dataset = torch.utils.data.TensorDataset(train_data)
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
    return loss, CEL, KLD

class VAE(nn.Module):
    def __init__(self, input_dim=768, h_dim=400):
        # 调用父类方法初始化模块的state
        super(VAE, self).__init__()
        self.output_dim = len(bert_tokenizer.vocab)
        self.input_dim = input_dim
        self.h_dim = h_dim
        

        # 编码器 ： [b, input_dim] => [b, z_dim]
        self.encoder = BertModel.from_pretrained(pretrain_path)
        #self.scaler = nn.Linear(input_dim, input_dim//4) 
        self.fc1 = nn.Linear(input_dim, input_dim)  # mu
        self.fc2 = nn.Linear(input_dim, h_dim)  # log_var
        self.fc3 = nn.Linear(input_dim, h_dim)
        # 解码器 ： [b, z_dim] => [b, input_dim]
        self.fc4 = nn.Linear(h_dim, input_dim)
        self.fc5 = nn.Linear(input_dim, self.output_dim)

    def forward(self, x):
        """
        向前传播部分, 在model_name(inputs)时自动调用
        :param x: the input of our training model [b, batch_size, 1, 28, 28]
        :return: the result of our training model
        """
        x = self.encoder(x)[0]

        # encoder
        mu, log_var = self.encode(x)
        # reparameterization trick
        sampled_z = self.reparameterization(mu, log_var)
        #print(sampled_z.shape)
        # decoder
        x_hat = self.decode(sampled_z)
        # reshape
        #print(x_hat.shape)
        #x_hat = x_hat.view(batch_size, 1, 28, 28)
        return x_hat, mu, log_var

    def encode(self, x):
        """
        encoding part
        :param x: input image
        :return: mu and log_var
        """
        #x = self.scaler(x)
        #print(x.shape)
        #batch_size = x.shape[0]
        #x = x.reshape(batch_size,-1)
        #print(x.shape)
        h = F.relu(self.fc1(x))
        # print(h.shape)
        mu = self.fc2(h)
        log_var = self.fc3(h)
        # print(mu.shape)
        # print(log_var.shape)
        return mu, log_var

    def reparameterization(self, mu, log_var):
        """
        Given a standard gaussian distribution epsilon ~ N(0,1),
        we can sample the random variable z as per z = mu + sigma * epsilon
        :param mu:
        :param log_var:
        :return: sampled z
        """
        sigma = torch.exp(log_var * 0.5)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps  # 这里的“*”是点乘的意思

    def decode(self, z):
        """
        Given a sampled z, decode it back to image
        :param z:
        :return:
        """
        h = F.relu(self.fc4(z))
        x_hat = torch.sigmoid(self.fc5(h))  # 图片数值取值为[0,1]，不宜用ReLU
        return x_hat

net = VAE().cuda()
optimizer = AdamW(net.parameters(),lr = 2e-5, eps = 1e-8)
index = 0
loss_sum = 0 

pbar = ProgressBar(n_total=len(train_iter), desc='Training')

def decode_test(model):
    z = torch.randn(1, 10 ,400).cuda() # 每一行是一个隐变量，总共有batch_size行
    # 对隐变量重构
    with torch.no_grad():
        random_res = model.decode(z)

    _, result = torch.max(random_res, 2)
    #print(result.shape)
    # exit()
    # exit()
    #predict = result.cpu().numpy().tolist()
    #print(predict)
    #for i in result:
    #print(result)
    result = result[0]
    #print(result.shape)
    tokens = bert_tokenizer.convert_ids_to_tokens(result)
    print(tokens)
    #exit()
    
decode_test(net)

for batch in train_iter:
    input_data = batch[0].cuda()
    x_hat, mu, log_var = net(input_data)
    loss, CEL, KLD = loss_function(x_hat,input_data,mu,log_var)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    index+=1
    loss_sum +=loss
    pbar(index-1, {'loss': loss_sum/index})
    if index % 20 == 0:
        decode_test(net)
        print(CEL,KLD,loss)
    # print(loss)
    # exit()

