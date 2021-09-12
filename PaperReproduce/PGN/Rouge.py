# test ='你好啊'
# print(" ".join(test))
# exit()
from config import dev_data_path,ref_path,output_path
import json
load_f = open(dev_data_path, 'r', encoding='utf-8')
temp = load_f.readlines()
#temp = temp[:10000]
write_f = open(ref_path,'w',encoding='utf-8')

print("dataNumber", len(temp))

refs = []
from tqdm import tqdm
for line in tqdm(temp):
    dic = json.loads(line)
    target = dic['summary']
    write_f.write(target+'\n')
    refs.append(" ".join(target))


import json
from rouge import Rouge
f = open(output_path,'r',encoding='utf-8')
lines = f.readlines()
assert len(lines) == len(refs)
lines = [" ".join(i.strip().replace('\n','').replace('</s>','')) for i in lines]


rouge = Rouge()
print(lines[0],refs[0])
print(rouge.get_scores(lines[0],refs[0]))
# print(rouge.get_scores("你好", "不好"))
scores = rouge.get_scores(lines, refs, avg=True)


print(scores)