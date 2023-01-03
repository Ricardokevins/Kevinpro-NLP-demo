import numpy as np
import torch
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')
V_set = torch.load('feature.pt')
score_set = torch.load('score.pt')
token_set = torch.load('tokens.pt')

# d = 64                              # 向量维度
# nb = 100000                         # 向量集大小
# nq = 10000                          # 查询次数
# np.random.seed(1234)                # 随机种子,使结果可复现
# xb = np.random.random((nb, d)).astype('float32')
# xb[:, 0] += np.arange(nb) / 1000.
# xq = np.random.random((nq, d)).astype('float32')
# xq[:, 0] += np.arange(nq) / 1000.
#  /home/data_ti4_d/wangjl/miniconda3/pkgs/libstdcxx-ng-11.2.0-h1234567_1/lib/libstdc++.so.6.0.29
import faiss

d = 1024
nlist = 100
m = 64
k = 10
lambda_weight = 0.6
quantizer = faiss.IndexFlatL2(d)    # 内部的索引方式依然不变
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
index.train(V_set)
index.add(V_set)


import numpy as np
import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration
# Load pre-trained model (weights)
device = torch.device("cpu")
pretrain_path = '/home/shesj/workspace/Data/PLM/linydub-bart-large-samsum'
model = BartForConditionalGeneration.from_pretrained(pretrain_path,output_hidden_states = True,output_attentions=True)
model.eval()
tokenizer = BartTokenizer.from_pretrained(pretrain_path)



from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    StoppingCriteriaList,
    MaxLengthCriteria,LogitsProcessorList,MinLengthLogitsProcessor,NoRepeatNGramLogitsProcessor,ForcedBOSTokenLogitsProcessor
)
import torch
#device = "cuda" if torch.cuda.is_available() else "cpu"
device = 'cpu'

import numpy as np
import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration

logits_processor = LogitsProcessorList([NoRepeatNGramLogitsProcessor(3),ForcedBOSTokenLogitsProcessor(2)])


def greedy_search(input_prompt):
    eos_token_id = model.config.eos_token_id
    decode_length = 100
    decoded_ids = torch.tensor([[2]]).to(device)
    for t in range(decode_length):
        encoded_src = tokenizer(
            [input_prompt],
            max_length=1024,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        #print(source)
        
        src_tokens = encoded_src['input_ids'].to(device)
        src_mask = encoded_src['attention_mask'].to(device)


        output = model(
            input_ids=src_tokens,
            attention_mask=src_mask,
            decoder_input_ids=decoded_ids
        )
        #logits = output.logits.view(-1, model.config.vocab_size)
        next_token_logits = output.logits[:, -1, :]
        
        
        next_tokens_scores = logits_processor(decoded_ids, next_token_logits)
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)
        decoded_ids = torch.cat([decoded_ids, next_tokens[:, None]], dim=-1)
        #print(next_tokens,eos_token_id)
        if next_tokens == eos_token_id and t>3:
            break
        #print(next_tokens)

    return decoded_ids

def greedy_search_withKNN(input_prompt):
    eos_token_id = model.config.eos_token_id
    decode_length = 100
    decoded_ids = torch.tensor([[2]]).to(device)
    for t in range(decode_length):
        encoded_src = tokenizer(
            [input_prompt],
            max_length=1024,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        #print(source)
        
        src_tokens = encoded_src['input_ids'].to(device)
        src_mask = encoded_src['attention_mask'].to(device)


        output = model(
            input_ids=src_tokens,
            attention_mask=src_mask,
            decoder_input_ids=decoded_ids
        )
        #logits = output.logits.view(-1, model.config.vocab_size)
        decoder_state = output.decoder_hidden_states[-1]
        #print(decoder_state.shape)
        D, I = index.search(decoder_state.reshape(-1,1024), k)      
        sft = nn.Softmax()
        D = sft(-torch.tensor(D)).reshape(-1)
        I = torch.tensor(I).reshape(-1)
        next_token_logits = output.logits[:, -1, :] * lambda_weight
        for sam_index in range(k):

            next_token_logits += score_set[I[sam_index]] * D[sam_index] * (1-lambda_weight)

        next_tokens_scores = logits_processor(decoded_ids, next_token_logits)
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)
        decoded_ids = torch.cat([decoded_ids, next_tokens[:, None]], dim=-1)
        #print(next_tokens,eos_token_id)
        if next_tokens == eos_token_id and t>3:
            break
        #print(next_tokens)

    return decoded_ids

with torch.no_grad():
    path = "/home/shesj/workspace/Data/Data/SAMSum/val.json"
    import json
    f = open(path,'r')
    data = json.load(f)
    data = data[10:]
    f.close()
    from tqdm import tqdm
    for i in tqdm(data):
        source = i['dialogue']
        ids = greedy_search_withKNN(source)
        hypo = tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(hypo)
        ids = greedy_search(source)
        hypo = tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(hypo)
        print(i['summary'])
        print("====================================================")
        # exit()
        # logits = return_state['logits'].view(-1, model.model.config.vocab_size)
        # decoder_state = return_state.decoder_hidden_states
        # decoder_state = decoder_state[-1].reshape(-1,1024)
        # print(V_set.shape)
        # print(decoder_state.shape)
        # D,I = index.search(decoder_state,4)
        # for i in I:
        #     for j in i:
        #         print(tokenizer.decode(token_set[j]))
        #     print('============================')
        #     #print(i)
        # exit()
# print(I)
# print(D)
# index.nprobe = 10                   # 与以前的方法相比
# D, I = index.search(xq, k)          # 检索
# print(I[-5:])