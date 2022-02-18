
import random

def random_sample(source,target,sample_num):
    idx = []
    while len(idx) < sample_num:
        new_idx = random.randint(0,len(source)-1)
        if new_idx not in idx:
            idx.append(new_idx)
    sampled_source = []
    sampled_target = []
    for i in idx:
        sampled_source.append(source[i])
        sampled_target.append(target[i])
    return sampled_source,sampled_target

def readFromPair(max_samples):
    f = open('./data/source.txt','r',encoding='utf-8')
    question = f.readlines()
    question = [i.replace('\n','') for i in question]
    f = open("./data/target.txt",'r',encoding = 'utf-8')
    answer = f.readlines()
    answer = [i.replace('\n','') for i in answer]
    if max_samples != -1:
        question,answer = random_sample(question,answer,max_samples)
    return question,answer
