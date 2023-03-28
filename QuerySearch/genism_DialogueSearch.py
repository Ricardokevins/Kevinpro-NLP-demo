#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
import math
from six import iteritems
from six.moves import xrange
import random
import os
random.seed(24)
# BM25 parameters.
PARAM_K1 = 1.5
PARAM_B = 0.75
EPSILON = 0.25
class BM25(object):
    def __init__(self, corpus):
        self.corpus_size = len(corpus)
        self.avgdl = sum(map(lambda x: float(len(x)), corpus)) / self.corpus_size
        self.corpus = corpus
        self.f = []
        self.df = {}
        self.idf = {}
        self.initialize()
    def initialize(self):
        for document in self.corpus:
            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.f.append(frequencies)
            for word, freq in iteritems(frequencies):
                if word not in self.df:
                    self.df[word] = 0
                self.df[word] += 1
        for word, freq in iteritems(self.df):
            self.idf[word] = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)

    def get_score(self, document, index, average_idf):
        score = 0
        for word in document:
            if word not in self.f[index]:
                continue
            idf = self.idf[word] if self.idf[word] >= 0 else EPSILON * average_idf
            score += (idf * self.f[index][word] * (PARAM_K1 + 1)
                      / (self.f[index][word] + PARAM_K1 * (1 - PARAM_B + PARAM_B * self.corpus_size / self.avgdl)))
        return score
    def get_scores(self, document, average_idf):
        scores = []
        for index in xrange(self.corpus_size):
            score = self.get_score(document, index, average_idf)
            scores.append(score)
        return scores
def get_bm25_weights(corpus):
    bm25 = BM25(corpus)
    average_idf = sum(map(lambda k: float(bm25.idf[k]), bm25.idf.keys())) / len(bm25.idf.keys())
    weights = []
    for doc in corpus:
        scores = bm25.get_scores(doc, average_idf)
        weights.append(scores)
    return weights


from gensim import corpora
from gensim.summarization import bm25
import os
import re


path = "/Users/sheshuaijie/Desktop/RearchSpace/Data/Data/SAMSum/train.json"
import json
f = open(path,'r')
data = json.load(f)
f.close()

import string
punctuation_string = string.punctuation + '’'
def re_punctuation(stri):
    for i in punctuation_string:
        stri = stri.replace(i, ' ').lower() 
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
    turns = [re_punctuation(i)for i in turns]
    #print(turns)
    turns = list(set(turns))
    # print(turns)
    # exit()
    turns = [i.split(" ") for i in turns]
    turns = [[j for j in i if len(j)!=0] for i in turns]
    
    
    all_tokens = []
    #biturn = return_biturn(turns)
    #Document_Features.extend(biturn)

    Document_Features.extend(turns)

import pickle

def bm25save(obj, fpath):
    f = open(fpath, 'wb')
    pickle.dump(obj, f)
    f.close()

def bm25load(fpath):
    f = open(fpath, 'rb')
    bm25Model = pickle.load(f)
    f.close()
    return bm25Model


bm25Model = bm25.BM25(Document_Features)


# input_query = 'are you in Warsaw? yes, just back!'
# #are you in Warsaw yes [UNK] back
# are you in [UNK]

input_query = "Yeah. I definitely prefer Lisbon Yeah me too"
# #Yeah I definitely [UNK] [UNK] Yeah me too
# yeah i definitely prefer [UNK]

#input_query = 'do you have Betty\'s number?'
# #do you have [UNK] s number Lemme check
# do you have [UNK] s number [UNK] check

#input_query = "Have you got any homework for tomorrow? no dad"
# #Have you got any homework for tomorrow no [UNK]
# have you got any [UNK] for tomorrow no [UNK]
# have you got any [UNK] for tomorrow

#input_query = " It's good for us, Vanessa and I are still on our way and Peter's stuck in a traffic"
# [UNK] s [UNK] for [UNK] [UNK] and I are [UNK] on our way and [UNK] s stuck in a traffic
# it s good for us [UNK] and i are still on our way and [UNK] s [UNK] in a [UNK]


# input_query = "Hey girls! Any plans for tonight? Want to drop by?"
# hey [UNK] any plans for [UNK] want to [UNK] by
# hey [UNK] any plans for tonight

from textblob import TextBlob
# import nltk
# nltk.download('brown')
while 1:
    input_query = input("Prompt: ")
    #blob = TextBlob(input_query)
    #blob.tags
    #print(blob.noun_phrases)

    input_query = re_punctuation(input_query)
    input_query_token = input_query.split(" ")
    input_query_token = [j for j in input_query_token if len(j)!=0]

    scores = bm25Model.get_scores(input_query_token)
    idf = bm25Model.idf
    #sc = [(i,round(idf.get(i,0),3)) for i in input_query_token]
    #print(sc)
    def getTopK(t,k=3000):
        #k = 3000
        #k = 3000
        max_index = []
        for _ in range(k):
            number = max(t)
            index = t.index(number)
            t[index] = 0
            max_index.append(index)
        return max_index

    index = getTopK(scores,k=1000)

    def process(index):
        onegram_freq = {}
        twogram_freq = {}
        for i in index:
            #print(data[i]['summary'])
            #print(" ".join(Document_Features[i]))
            TwoGram = []
            for _ in range(len(Document_Features[i])-1):
                twogram = Document_Features[i][_] + " " + Document_Features[i][_+1]
                twogram_freq[twogram] = twogram_freq.get(twogram,0) + 1
            
            for _ in range(len(Document_Features[i])):
                onegram = Document_Features[i][_]
                onegram_freq[onegram] = onegram_freq.get(onegram,0) + 1
            #print(pre_score_list[i])


        one_gram_sort_result = sorted(onegram_freq.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)[:input_query_token.__len__()*1]
        two_gram_sort_result = sorted(twogram_freq.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)[:input_query_token.__len__()*1]

        freq_one_gram = [i[0] for i in one_gram_sort_result]
        freq_two_gram_count = [i[0] for i in two_gram_sort_result]
        freq_two_gram1 = [i.split(" ")[0] for i in freq_two_gram_count]
        freq_two_gram2 = [i.split(" ")[1] for i in freq_two_gram_count]
        freq_two_gram = freq_two_gram1 + freq_two_gram2
        freq_two_gram = freq_two_gram[:input_query_token.__len__()*1]

        pattern = []
        pattern2 = []
        #print(blob.tags)
        print(one_gram_sort_result)
        print(freq_two_gram_count)
        for i in input_query_token:
            #if i in freq_one_gram or t[1] not in ['NN','NNS','NNP','NNPS']:
            if i in freq_one_gram:
                pattern.append(i)
            else:
                pattern.append('_'*len(i))
            if i in freq_two_gram:
            #if i in freq_two_gram or t[1] not in ['NN','NNS','NNP','NNPS']:
                pattern2.append(i)
            else:
                pattern2.append('_'*len(i))

            
        print("============================================")
        print(" ".join(input_query_token))
        print(" ".join(pattern))
        print(" ".join(pattern2))
        print("============================================")

    def rerank():
        result = []
        for q in input_query_token:
            input_truncated = input_query_token.copy()
            input_truncated.remove(q)
            scores = bm25Model.get_scores(input_truncated)
            index = getTopK(scores,30)

            
        
            onegram_freq = {}
            for i in index:             
                for _ in range(len(Document_Features[i])):
                    onegram = Document_Features[i][_]
                    onegram_freq[onegram] = onegram_freq.get(onegram,0) + 1
            #print(pre_score_list[i])


            one_gram_sort_result = sorted(onegram_freq.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)[:input_query_token.__len__()*2]
            freq_one_gram = [i[0] for i in one_gram_sort_result]
            print(q,one_gram_sort_result)
            if q in freq_one_gram:
                result.append(q)
            else:
                result.append('_'*len(q))
        print(" ".join(result))

    
    #rerank(index)
    process(index)

    #rerank()
    
    
    # tail_index = index[:20]
    # for i in tail_index:
    #     print(" ".join(Document_Features[i]))

    # Oh no, I'm staying in Warsaw till Saturday only :(
    # Oh no, I'm staying in till Saturday only :(
    # Oh no, I'm in Warsaw utill Saturday only :(
    # Oh no, I'm staying in Warsaw utill  only :(
    # Oh no, I'm in Warsaw utill  only :(