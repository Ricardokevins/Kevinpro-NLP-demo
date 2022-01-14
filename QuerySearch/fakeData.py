base_path = "/root/SheShuaijie/workspace/Robust/rawdata/"
target_path = "/root/SheShuaijie/workspace/Robust/data/"
#data = "qags_cnn_train.json"
data = 'data_rank19.json'
target_data = 'data-dev.jsonl'
import json
from mlmtest import attack
from tqdm import tqdm

def convert_qags(file_name):
    f = open(base_path + file_name,'r',encoding='utf-8')
    lines = f.readlines()
    
    convert_samples = []
    index = 0
    for i in lines:
        i = i.strip()
        data_dict = json.loads(i)
        new_data_dict = {}
        new_data_dict['text'] = data_dict['text']
        new_data_dict['id'] = index
        index += 1
        new_data_dict['claim'] = data_dict['claim']
        if data_dict['label'] == 1:
            new_data_dict['label'] = "CORRECT"
        else:
            new_data_dict['claim'] = new_data_dict['claim'] + " " + new_data_dict['text'].split(".")[0] + "."
            new_data_dict['label'] = "INCORRECT"
        convert_samples.append(new_data_dict)

    with open(target_path+target_data, "w",encoding = 'utf-8') as fd:
        for example in convert_samples:
            fd.write(json.dumps(example, ensure_ascii=False) + "\n")
        
# convert_qags(data)



def convert_qags_targeted(file_name):
    preds = []
    f = open("/root/SheShuaijie/workspace/Robust/data/eval_result.txt",'r',encoding='utf-8')
    preds = f.readlines()
    preds = [int(i.strip().split(' ')[0]) for i in preds]
    f = open(base_path + file_name,'r',encoding='utf-8')
    lines = f.readlines()
    
    convert_samples = []
    index = 0
    for i in lines:
        i = i.strip()
        data_dict = json.loads(i)
        new_data_dict = {}
        new_data_dict['text'] = data_dict['text']
        new_data_dict['id'] = index
        
        new_data_dict['claim'] = data_dict['claim']
        if data_dict['label'] == 1:
            new_data_dict['label'] = "CORRECT"
        else:
            if preds[index] == 1:
                new_data_dict['claim'] = new_data_dict['claim'] + " " + new_data_dict['text'].split(".")[0] + "."
            new_data_dict['label'] = "INCORRECT"
        convert_samples.append(new_data_dict)
        index += 1

    with open(target_path+target_data, "w",encoding = 'utf-8') as fd:
        for example in convert_samples:
            fd.write(json.dumps(example, ensure_ascii=False) + "\n")

#convert_qags_targeted(data)

def convert_rank19(file_name):
    f = open(base_path + file_name,'r',encoding='utf-8')
    lines = f.readlines()
    
    convert_samples = []
    index = 0
    for i in tqdm(lines):
        i = i.strip()
        data_dict = json.loads(i)
        new_data_dict = {}
        #new_data_dict['text'] = "The 28 minute movie in which a woman dressed as a school girl, has sex with a man in a vintage train carriage was filmed on board the epping ongar. Locals in the area were shocked to learn the location, a favourite with families and children, had been rented out by its bosses to american adult film company brazzers. The storyline has been viewed more than 235,000 times." + data_dict['src'] + "Aidy boothroyd will be the man in charge of the under 20s this time. Toulon tournament runs from may 27 to june 7. Gareth southgate's squad finished fourth last may."
        new_data_dict['text'] = data_dict['src']
        new_data_dict['id'] = index
        index += 1
        #new_data_dict['claim'] = data_dict['sys_summs']['correct']['sys_summ']
        new_data_dict['claim'] = attack(data_dict['sys_summs']['correct']['sys_summ'],0.2)
        new_data_dict['label'] = "CORRECT"
        convert_samples.append(new_data_dict)

        new_data_dict = {}
        #new_data_dict['text'] = "The 28 minute movie in which a woman dressed as a school girl, has sex with a man in a vintage train carriage was filmed on board the epping ongar. Locals in the area were shocked to learn the location, a favourite with families and children, had been rented out by its bosses to american adult film company brazzers. The storyline has been viewed more than 235,000 times." + data_dict['src'] + "Aidy boothroyd will be the man in charge of the under 20s this time. Toulon tournament runs from may 27 to june 7. Gareth southgate's squad finished fourth last may."
        new_data_dict['text'] = data_dict['src']
        new_data_dict['id'] = index
        index += 1
        #new_data_dict['claim'] = data_dict['sys_summs']['incorrect']['sys_summ'] + " " + new_data_dict['text']
        new_data_dict['claim'] = data_dict['sys_summs']['incorrect']['sys_summ']
        new_data_dict['label'] = "INCORRECT"
        convert_samples.append(new_data_dict)

    with open(target_path+target_data, "w",encoding = 'utf-8') as fd:
        for example in convert_samples:
            fd.write(json.dumps(example, ensure_ascii=False) + "\n")

    
convert_rank19(data)