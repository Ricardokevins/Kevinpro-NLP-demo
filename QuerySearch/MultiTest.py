import json
from itertools import combinations
from rouge import Rouge
import threading
import time
import math
import datetime
import multiprocessing as mp
from rouge import Rouge
import copy
import json
import random
random.seed(413)
rouge = Rouge()

def get_score(hyp,ref):
    try:
        temp_rouge = rouge.get_scores(hyp, ref)
        cur_score = (temp_rouge[0]["rouge-1"]['f'] + temp_rouge[0]["rouge-2"]['f'] + temp_rouge[0]["rouge-l"]['f'])/3
    except :
        cur_score = 0
    return cur_score

def get_oracle(sent_list,summary):
    Chosen_idx = []
    best_score = 0
    cal_count = 0 
    while 1:
        best_choice = -1
        best_sub_score = 0
        for i in range(len(sent_list)):
            if i not in Chosen_idx and len(sent_list[i]) != 0 :
                cal_count += 1
                temp_chosen = copy.deepcopy(Chosen_idx)
                temp_chosen.append(i)
                temp_chosen_sents = [sent_list[i] for i in temp_chosen]
                #print(temp_chosen)
                #print(temp_chosen_sents)
                cur_score = get_score(" ".join(temp_chosen_sents),summary)
                cur_sub_score = cur_score - best_score
                if cur_sub_score > best_sub_score:
                    best_sub_score = cur_sub_score
                    best_choice = i
        if best_choice == -1:
            break
        Chosen_idx.append(best_choice)
        best_sents = [sent_list[i] for i in Chosen_idx]
        best_score = get_score(" ".join(best_sents),summary)

    best_sents = [sent_list[i] for i in Chosen_idx]
    #print(len(sent_list))
    #print(len(best_sents))
    #print(cal_count)
    try:
        temp_rouge = rouge.get_scores(" ".join(best_sents), summary)
    except :
        return 0,0,0
    
    return temp_rouge[0]["rouge-1"]['f'],temp_rouge[0]["rouge-2"]['f'],temp_rouge[0]["rouge-l"]['f']

def Function(name,json_data):
    result = {}
    result['r1'] = []
    result['r2'] = []
    result['rl'] = []
    for i in range(len(json_data)):
        doc = json_data[i]['doc']
        summary = json_data[i]['summary']
        r1,r2,rl = get_oracle(doc,summary)
        result['r1'].append(r1)
        result['r2'].append(r2)
        result['rl'].append(rl)


    return result

def prepare_data():
    input_data = []
    f = open('test.extract.source','r',encoding = 'utf-8')
    f2 = open('test.target','r',encoding = 'utf-8')
    f3 = open('QueryResult.txt','r',encoding = 'utf-8')
    query = f3.readlines()
    query = [[int (j) for j in i.strip().split()] for i in query]


    summarys = f2.readlines()
    summarys = [i.strip() for i in summarys]
    import random
    data_index = []
    while len(data_index) < 2000:
        random_index = random.randint(0,len(summarys)-1)
        if random_index not in data_index:
            data_index.append(random_index)

    print(data_index[:10])
    assert data_index[0] == 10455
    lines = f.readlines()
    ftrain = open('train.extract.source','r',encoding = 'utf-8')
    assist_lines = ftrain.readlines()



    for i in range(len(lines)):
        data = lines[i].strip()
        data_dict = json.loads(data)
        doc = data_dict['text']
        
        for j in query[i][:1]:
            assist = assist_lines[j].strip()
            assist_dict = json.loads(assist)
            assist_doc = assist_dict['text']
            doc = assist_doc + doc
        temp_data = {}
        temp_data['doc'] = doc
        temp_data['summary'] = summarys[data_index[i]]

        input_data.append(temp_data)
    return input_data

def extract_result(results):
    total_samples = 0
    Sum1 = 0
    Sum2 = 0
    SumL = 0
    for i in results:
        total_samples += len(i['r1'])
        Sum1 += sum(i['r1'])
        Sum2 += sum(i['r2'])
        SumL += sum(i['rl'])
    
    print(total_samples)
    print(Sum1/total_samples)
    print(Sum2/total_samples)
    print(SumL/total_samples)

def multi_process_tag(target_data):
    num_cores = int(mp.cpu_count())
    print("本地计算机有: " + str(num_cores) + " 核心")
    pool = mp.Pool(num_cores)
    
    param_dict = {}
    start = 0
    end =  len(target_data)
    step = int((end - start)/num_cores)
    print("per Task Step: ",step)
    for i in range(num_cores):
        param_dict['task{}'.format(i)]= target_data[start:start+step]
        start = start+step
    param_dict['task{}'.format(num_cores)]= target_data[start:]
    start_t = datetime.datetime.now()
    results = [pool.apply_async(Function, args=(name, param)) for name, param in param_dict.items()]
    results = [p.get() for p in results]
    
    end_t = datetime.datetime.now()
    elapsed_sec = (end_t - start_t).total_seconds()
    print("多进程计算 共消耗: " + "{:.2f}".format(elapsed_sec) + " 秒")
    return results

if __name__ ==  '__main__':
    data = prepare_data()
    results = multi_process_tag(data)
    extract_result(results)