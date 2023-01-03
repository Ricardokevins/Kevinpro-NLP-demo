from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    StoppingCriteriaList,
    MaxLengthCriteria,LogitsProcessorList,MinLengthLogitsProcessor,NoRepeatNGramLogitsProcessor,ForcedBOSTokenLogitsProcessor
)
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("/Users/sheshuaijie/Desktop/RearchSpace/Data/PLM/linydub-bart-large-samsum")
model = AutoModelForSeq2SeqLM.from_pretrained("/Users/sheshuaijie/Desktop/RearchSpace/Data/PLM/linydub-bart-large-samsum")
input_prompt = "Aude: Hi Susie, how is Ted this morning? Did you find plasters?\nSusie: yes. He kept them till this morning after his shower.\nAude: he must look sexy whith them... lol\nSusie: a  memory from  Poland!"

inputs = tokenizer([input_prompt], max_length=1024, return_tensors="pt")



import numpy as np
import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration


loss_fct = nn.NLLLoss(reduction='none', ignore_index=model.config.pad_token_id)
lsm = nn.LogSoftmax(dim=1)
def get_candidate(logits):
    prob_ = lsm(logits)
    values, indices = prob_.topk(10, dim=1, largest=True, sorted=True)
    return indices

def de_tokenize(token_index):
    """
    @description  : Use tokenizer to decode the token_index
    ---------
    @param  : 
        tokenindex: tensor
    -------
    @Returns  : token_list
    -------
    """
    token_list = []
    for j in token_index:
        token_list.append(tokenizer._convert_id_to_token(j.cpu().numpy().tolist()))
    filtered_token_list = []
    for i in token_list:
        filtered_token_list.append(tokenizer.convert_tokens_to_string([i]))
    return filtered_token_list
        

decode_length = 100
# tgt_list=['']
decoded_ids = torch.tensor([[2]]).to(device)
logits_processor = LogitsProcessorList([NoRepeatNGramLogitsProcessor(3),ForcedBOSTokenLogitsProcessor(2)])
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

   
    # src_tokens = src_tokens[:,1:-1]
    # src_mask = src_mask[:,1:-1]
    # summary_ids = model.generate(src_tokens, num_beams=1, min_length=0, max_length=100)
    # print(tokenizer.batch_decode(summary_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0])
    # print(summary_ids)
    # exit()
    #exit()
    #print(tgt_tokens)
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
    # print(candidate_decode[:,0])
    # candidate_result = de_tokenize(candidate_decode[:,0])
    # candidate_result2 = de_tokenize(candidate_decode[:,1])
    # print(de_tokenize(tgt_tokens.reshape(-1)))
    # print(candidate_result)
    # print(candidate_result2)
    # decode_result = candidate_result[-1]
    # if decode_result == "</s>":
    #     break
    # tgt_list[0] = tgt_list[0] + decode_result
    # print(tgt_list)
    #print(decoded_ids)

   