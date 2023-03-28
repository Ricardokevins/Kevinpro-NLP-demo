from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
from transformers import GPTJForCausalLM
import json
import torch
import time
import os
from config import LLM_Config
from utils import load_model_tokenizer,return_log_writer
# CUDA_VISIBLE_DEVICES=3,5 deepspeed --num_gpus=2 run_LLM.py  --do_eval --deepspeed ds_config.json
# deepspeed --include="localhost:3,5" run_LLM.py  --do_eval --deepspeed ds_config.json
# deepspeed --include="localhost:0" run_LLM.py  --do_eval --deepspeed ds_config.json

name = input("Model name: ")
if name not in LLM_Config:
    print("Error")

model,tokenizer = load_model_tokenizer(name)
logger = return_log_writer(name)

while 1:
    input_text = input("Prompt: ")
    if len(input_text) == 0:
        continue
    if input_text[0].strip() == "":
        continue
    if input_text[0] == "+":#Special Prompt to Decode Files    
        f = open(input_text[1:]+'.json','r')
        data = json.load(f)
        #data = data[:10]
        from tqdm import tqdm 
        for i in tqdm(data):
            text = i['input']
            inputs = tokenizer.encode(text, return_tensors="pt",max_length=2048,truncation=True)
            inputs = inputs.cuda()
            decode_length_constrain = list(inputs.shape)[1] + 150
            # if list(inputs.shape)[1] > 600:
            #     decode_length_constrain = list(inputs.shape)[1] + 100
            # else:
            #     decode_length_constrain = 1024
            with torch.no_grad():
                outputs = model.generate(inputs,max_length=decode_length_constrain)
                #outputs = model.generate(inputs,max_length=1024)
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            i['target'] = generated

        f = open(input_text[1:]+'_decoded_'+ name + '.json','w')
        json.dump(data,f,indent=4)   
        f.close()         
    else:
        inputs = tokenizer.encode(input_text, return_tensors="pt",max_length=1024,truncation=True)
        inputs = inputs.cuda()
        with torch.no_grad():
            outputs = model.generate(inputs,max_length=1024)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(name + ": ",generated)
        data = {'input':input_text,'generated':generated}
        data = json.dumps(data, indent=4)
        logger.write(data+'\n')
        logger.flush()
        print("================================================================")


