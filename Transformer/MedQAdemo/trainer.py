from torch.utils.data import Dataset
from util import *
class CharDataset(Dataset):
    def __init__(self):
        question,answer = readFromPair()
        if os.path.exists('dict.txt') == False:
            self.build_dict(question, answer)
        else:
            self.load_dict()
    def build_dict(self,source,target):
        self.word2id = {'[PAD]':0,'[BOS]':1,'[EOS]':2}
        id = 3
        for s,t in zip(source,target):
            for i in s:
                if i not in self.word2id:
                    self.word2id[i] = id 
                    id += 1
            for i in t:
                if i not in self.word2id:
                    self.word2id[i] = id 
                    id += 1
            if id > 30000:
                break
        
        self.id2word = {}
        for i in self.word2id:
            self.id2word[self.word2id[i]] = i
        
        f = open('dict.txt','w',encoding='utf-8')
        for i in self.word2id:
            f.write(i + " " + self.word2id[i] + "\n")
        return

    def __len__(self):
        return 

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        return 