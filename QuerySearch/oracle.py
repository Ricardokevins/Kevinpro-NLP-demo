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


r1 = 0 
r2 = 0
rl = 0
lines = f.readlines()

from tqdm import tqdm 
for i in tqdm(range(len(lines))):
    data = lines[i].strip()
    data_dict = json.loads(data)
    doc = data_dict['text']

    for j in query[i][:3]:
        assist = lines[j].strip()
        assist_dict = json.loads(assist)
        assist_doc = assist_dict['text']
        doc = assist_doc + doc

    r1_s,r2_s,rl_s = get_oracle(doc,summarys[data_index[i]])
    r1 += r1_s 
    r2 += r2_s 
    rl += rl_s

print("ROUGE Score : ROUGE1: {} ROUGE2: {} ROUGEL: {}".format(r1/len(lines), r2/len(lines), rl/len(lines)))