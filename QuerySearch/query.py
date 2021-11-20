path = 'train.target'

# Follow implement in https://blog.csdn.net/chaojianmo/article/details/105143657

# path = 'train.source'
from tqdm import tqdm
import numpy as np
class TF_IDF_Model(object):
    def __init__(self, documents_list):
        self.documents_list = documents_list
        self.documents_number = len(documents_list)
        self.tf = []
        self.idf = {}
        self.init()
 
    def init(self):
        df = {}
        for document in tqdm(self.documents_list):
            temp = {}
            for word in document:
                temp[word] = temp.get(word, 0) + 1/len(document)
            self.tf.append(temp)
            for key in temp.keys():
                df[key] = df.get(key, 0) + 1
        for key, value in df.items():
            self.idf[key] = np.log(self.documents_number / (value + 1))
 
    def get_score(self, index, query):
        score = 0.0
        for q in query:
            if q not in self.tf[index]:
                continue
            score += self.tf[index][q] * self.idf[q]
        return score
 
    def get_documents_score(self, query):
        score_list = []
        best_score = -1
        best_result = 0
        result_list = []
        for i in tqdm(range(self.documents_number)):
            cur_score = self.get_score(i, query)
            score_list.append(cur_score)
            if best_score < cur_score:
                best_score = cur_score
                # best_result = i
                #print(best_score)
                #print(self.documents_list[i])
                result_list.append(self.documents_list[i])
        for i in result_list[-5:]:
            print(" ".join(i))
        #print(result_list[:3])
        return score_list


import numpy as np
from collections import Counter
 
 
class BM25_Model(object):
    def __init__(self, documents_list, k1=2, k2=1, b=0.5):
        self.documents_list = documents_list
        self.documents_number = len(documents_list)
        self.avg_documents_len = sum([len(document) for document in documents_list]) / self.documents_number
        self.f = []
        self.idf = {}
        self.k1 = k1
        self.k2 = k2
        self.b = b
        self.init()
 
    def init(self):
        df = {}
        for document in self.documents_list:
            temp = {}
            for word in document:
                temp[word] = temp.get(word, 0) + 1
            self.f.append(temp)
            for key in temp.keys():
                df[key] = df.get(key, 0) + 1
        for key, value in df.items():
            self.idf[key] = np.log((self.documents_number - value + 0.5) / (value + 0.5))
 
    def get_score(self, index, query):
        score = 0.0
        document_len = len(self.f[index])
        qf = Counter(query)
        for q in query:
            if q not in self.f[index]:
                continue
            score += self.idf[q] * (self.f[index][q] * (self.k1 + 1) / (
                        self.f[index][q] + self.k1 * (1 - self.b + self.b * document_len / self.avg_documents_len))) * (
                                 qf[q] * (self.k2 + 1) / (qf[q] + self.k2))
 
        return score
 
    def get_documents_score(self, query):
        score_list = []
        best_score = -1
        best_result = 0
        result_list = []
        for i in tqdm(range(self.documents_number)):
            cur_score = self.get_score(i, query)
            score_list.append(cur_score)
            if best_score < cur_score:
                best_score = cur_score
                # best_result = i
                #print(best_score)
                #print(self.documents_list[i])
                result_list.append(self.documents_list[i])
        for i in result_list[-5:]:
            print(" ".join(i))

f = open(path,'r',encoding='utf-8')
lines = f.readlines()
lines = [i.strip().split() for i in lines]

input_query = "The Spanish football league has asked Uefa to investigate into whether Manchester City have broken financial fair play rules.".split()
#input_query = lines[-1]
model1 = TF_IDF_Model(lines)
model1.get_documents_score(input_query)

model2 = BM25_Model(lines)
model2.get_documents_score(input_query)