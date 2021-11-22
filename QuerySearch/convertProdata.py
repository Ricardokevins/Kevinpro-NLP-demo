from rouge import Rouge
import copy
import json
import random
random.seed(413)
rouge = Rouge()

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

fwrite = open("test1Assist.source",'w',encoding = 'utf-8')
from tqdm import tqdm 
for i in tqdm(range(len(lines))):
    data = lines[i].strip()
    data_dict = json.loads(data)
    doc = data_dict['text']

    for j in query[i][:1]:
        assist = assist_lines[j].strip()
        assist_dict = json.loads(assist)
        assist_doc = assist_dict['text']
        doc =  doc + assist_doc 
    proData = " ".join(doc)
    fwrite.write(proData + '\n')




    

