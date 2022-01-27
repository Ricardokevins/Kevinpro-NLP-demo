
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

def readFromPair():
    f = open('./data/question.txt','r',encoding='utf-8')
    question = f.readlines()
    question = [i.replace('\n','') for i in question]
    f = open("./data/answer.txt",'r',encoding = 'utf-8')
    answer = f.readlines()
    answer = [i.replace('\n','') for i in answer]
    #question,answer = random_sample(question,answer,10000)
    return question,answer
