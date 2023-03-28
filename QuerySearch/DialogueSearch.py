#path = 'train.target'

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
        for document in tqdm(self.documents_list):
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
        for i in range(self.documents_number):
            cur_score = self.get_score(i, query)
            score_list.append(cur_score)
        #     if best_score < cur_score:
        #         best_score = cur_score
        #         # best_result = i
        #         #print(best_score)
        #         #print(self.documents_list[i])
        #         result_list.append(self.documents_list[i])
        # for i in result_list[-3:]:
        #     print(" ".join(i))
        return score_list

path = "/Users/sheshuaijie/Desktop/RearchSpace/Data/Data/SAMSum/train.json"
import json
f = open(path,'r')
data = json.load(f)
f.close()

import string
punctuation_string = string.punctuation
def re_punctuation(stri):
    for i in punctuation_string:
        stri = stri.replace(i, ' ')
    return stri

def return_biturn(turns):
    biturns = []
    for i in range(0,len(turns)-1):
        biturns.append(turns[i]+turns[i+1])
    return biturns

Document_Features = []
for i in range(len(data)):
    dialogue = data[i]['dialogue']
    turns = dialogue.split('\n')
    turns = [i.strip()[len(i.split(":")[0])+2:] for i in turns]
    turns = [re_punctuation(i) for i in turns]

    turns = [i.split(" ") for i in turns]
    turns = [[j for j in i if len(j)!=0] for i in turns]
    # print(turns)
    # exit()
    all_tokens = []
    biturn = return_biturn(turns)
    Document_Features.extend(biturn)
    # for t in turns:
    #     all_tokens.extend(t)
    #     Document_Features.append(t)
    # print(all_tokens)
    # exit()
    #Document_Features.append(all_tokens)
model = TF_IDF_Model(Document_Features)
model2 = BM25_Model(Document_Features)

def getTopK(t):
    k = 30
    max_index = []
    for _ in range(k):
        number = max(t)
        index = t.index(number)
        t[index] = 0
        max_index.append(index)
    return max_index


f = open("/Users/sheshuaijie/Desktop/RearchSpace/Data/Data/SAMSum/val.json",'r')
testdata = json.load(f)
dialogue = testdata[1]['dialogue']
turns = dialogue.split('\n')
turns = [i.strip()[len(i.split(":")[0])+2:] for i in turns]
turns = [i.split(" ") for i in turns]
input_query = []
# for i in range(len(turns)):
#     input_query = turns[i]
#     # for t in turns:
#     #     input_query.extend(t)

#     print(" ".join(input_query))
#     print("=====================================================")
#     score_list = model2.get_documents_score(input_query)
#     pre_score_list = score_list.copy()
#     best = getTopK(score_list)
#     for i in best:
#         #print(data[i]['summary'])
#         print(" ".join(Document_Features[i]))
#         print(pre_score_list[i])
#input_query = "Do you want some? Sure".split(" ")
#input_query = "Have you got any homework for tomorrow? no dad".split(" ")
#input_query = "What did you plan on doing?".split(" ")
input_query = "are you in Warsaw? yes, just back!"
#input_query = 'do you have Betty\'s number? Lemme check'.split(" ")
#input_query = " It's good for us, Vanessa and I are still on our way and Peter's stuck in a traffic".split(" ")
#input_query = "can you take your dog away before i come?".split(" ")
input_query = "Yeah. I definitely prefer Lisbon Yeah me too"

input_query = re_punctuation(input_query)
input_query_token = input_query.split(" ")
input_query_token = [j for j in input_query_token if len(j)!=0]
print(input_query)
score_list = model2.get_documents_score(input_query_token)
#score_list = model2.get_documents_score(input_query)
pre_score_list = score_list.copy()
best = getTopK(score_list)


stop_words = []
f = open('/Users/sheshuaijie/Downloads/stop_words_english.txt','r')
lines = f.readlines()
stop_words = [i.strip() for i in lines]

onegram_freq = {}
twogram_freq = {}
for i in best:
    #print(data[i]['summary'])
    print(" ".join(Document_Features[i]))
    TwoGram = []
    for _ in range(len(Document_Features[i])-1):
        twogram = Document_Features[i][_] + " " + Document_Features[i][_+1]
        twogram_freq[twogram] = twogram_freq.get(twogram,0) + 1
    
    for _ in range(len(Document_Features[i])):
        #twogram = Document_Features[i][_] + " " + Document_Features[i][_+1]
        onegram = Document_Features[i][_]
        onegram_freq[onegram] = onegram_freq.get(onegram,0) + 1
    #print(pre_score_list[i])


one_gram_sort_result = sorted(onegram_freq.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)[:30]
two_gram_sort_result = sorted(twogram_freq.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)[:30]

freq_one_gram = [i[0] for i in one_gram_sort_result]
pattern = []
for i in input_query_token:
    if i in freq_one_gram:
        pattern.append(i)
    else:
        pattern.append('[UNK]')


print("============================================")
print(input_query)
print(" ".join(pattern))
# for i,j in zip(one_gram_sort_result,two_gram_sort_result):
#     # if i[0] in stop_words:
#     #     continue
#     print(i,j)
# for i in tqdm(range(len(lines))):
#     data_dict = json.loads(lines[i])
#     score_list = model2.get_documents_score(data_dict['feature'])
    
#     score_list[i] = -1
#     best = getTopK(score_list)
#     best = [str(i) for i in best]
#     fout.write(" ".join(best) + '\n')

# input_query = lines[-1]
# model1 = TF_IDF_Model(lines)
# model1.get_documents_score(input_query)

# model2 = BM25_Model(lines)
# model2.get_documents_score(input_query)