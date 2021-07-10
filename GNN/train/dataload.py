import torch
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
from torch_geometric.data import Data
import transformer
test_data='tests.txt'
train_data='trains.txt'
max_length = 64

def get_dataset(corpur_path):
    texts = []
    labels = []
    
    with open(corpur_path) as f:
        li = []
        while True:
            content = f.readline().replace('\n', '')
            if not content:              #为空行，表示取完一次数据（一次的数据保存在li中）
                if not li:               #如果列表也为空，则表示数据读完，结束循环
                    break
                label = li[0][10]
                text = li[1][6:-7]
                texts.append(text)
                labels.append(int(label))
                li = []
            else:
                li.append(content)       #["<Polarity>标签</Polarity>", "<text>句子内容</text>"]

    return texts, labels

def build_tokenizer():
    model = transformer.Transformer(max_length=max_length)
    text, _ = get_dataset(train_data)
    tokenizer = model.get_tokenzier(text)
    return tokenizer

tokenzier = build_tokenizer()


def get_dict_size():
    return len(tokenzier.word2idx)

def load_train():
    sents = []
    bert_sent = []
    text,labels = get_dataset(train_data)
    print(len(text))
    for i in text:
        sents.append(tokenzier.encode(i))

    return sents,labels

def load_test():
    sents = []
    bert_sent = []
    text,labels = get_dataset(test_data)
    for i in text:
        sents.append(tokenzier.encode(i))
    return sents,labels
    

def convert_sent2graph(sent):
    node = []
    edge = []
    step = [-3,-2,-1]
    for i in range(len(sent)):
        if sent[i]!=0:
            node.append(sent[i])
            for j in step:
                if i+j>=0 and i+j<len(sent) and sent[i+j]!=0:
                    edge.append([i,i+j])
                    edge.append([i+j,i])
    #node = range(2462)
    return node,edge

import torch.nn as nn
vocab = get_dict_size()
embedding = nn.Embedding(vocab, 256)

class TrainDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(TrainDataset, self).__init__(root, transform, pre_transform)
        
        self.data, self.slices = torch.load(self.processed_paths[0])
        

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['train.pt']

    def download(self):
        pass
    
    def process(self):
        
        data_list = []
        sent,label = load_train()

        # process by session_id
        # grouped = df.groupby('session_id')
        

        for s,l in tqdm(zip(sent,label)):
            node,edge = convert_sent2graph(s)
            node = torch.tensor(node)
            edge = torch.tensor(edge).transpose(0,1)
            #print(node.shape)
            #print(edge.shape)
            y = torch.tensor([l])
            
            node_features = embedding(node)
            #print(node_features.shape)
            data = Data(x=node_features, edge_index=edge, y=y)
            data_list.append(data) 
            # print(node_features.shape)
            # print(edge.shape)
            # print(y)
        
        
        data, slices = self.collate(data_list)
        # print(slices)
        torch.save((data, slices), self.processed_paths[0])

class TestDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(TestDataset, self).__init__(root, transform, pre_transform)
        
        self.data, self.slices = torch.load(self.processed_paths[0])
        

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['test.pt']

    def download(self):
        pass
    
    def process(self):
        
        data_list = []
        sent,label = load_test()

        # process by session_id
        # grouped = df.groupby('session_id')
        

        for s,l in tqdm(zip(sent,label)):
            node,edge = convert_sent2graph(s)
            node = torch.tensor(node)
            edge = torch.tensor(edge).transpose(0,1)
            #print(node.shape)
            #print(edge.shape)
            y = torch.tensor([l])
            
            node_features = embedding(node)
            #print(node_features.shape)
            data = Data(x=node_features, edge_index=edge, y=y)
            data_list.append(data) 
            # print(node_features.shape)
            # print(edge.shape)
            # print(y)
        
        
        data, slices = self.collate(data_list)
        # print(slices)
        torch.save((data, slices), self.processed_paths[0])


        


        