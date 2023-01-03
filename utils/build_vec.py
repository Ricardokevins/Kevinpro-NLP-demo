import numpy as np
import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration
# Load pre-trained model (weights)
device = torch.device("cpu")
model = BartForConditionalGeneration.from_pretrained('/Users/sheshuaijie/Desktop/RearchSpace/Data/PLM/linydub-bart-large-samsum',output_hidden_states = True,output_attentions=True)
model.eval()
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BartTokenizer.from_pretrained('/Users/sheshuaijie/Desktop/RearchSpace/Data/PLM/linydub-bart-large-samsum')

Feature = []
Score_distribution = []
tokens = []
with torch.no_grad():
    path = "/Users/sheshuaijie/Desktop/workspace/Data/Data/SAMSum/train.json"
    import json
    f = open(path,'r')
    data = json.load(f)
    f.close()
    from tqdm import tqdm
    for i in tqdm(data):
        source = i['dialogue']
        target = i['summary']
        src_encoded = tokenizer(
                            [source],
                            max_length=1024,
                            truncation=True,
                            padding=True,
                            return_tensors='pt'
                        )
        src_tokens = src_encoded['input_ids'].to(device)
        src_attn_mask = src_encoded['attention_mask'].to(device)

        tgt_encoded = tokenizer(
                            [target],
                            max_length=1024,
                            truncation=True,
                            padding=True,
                            return_tensors='pt'
                        )
        tgt_tokens = tgt_encoded['input_ids'].to(device)
        tgt_attn_mask = tgt_encoded['attention_mask'].to(device)

        return_state = model(
            input_ids=src_tokens,
            attention_mask=src_attn_mask,
            labels=tgt_tokens
        )
        logits = return_state['logits'].view(-1, model.model.config.vocab_size)
        decoder_state = return_state.decoder_hidden_states
        Feature.append(decoder_state[-1].reshape(-1,1024))
        Score_distribution.append(logits)
        tokens.append(tgt_tokens.reshape(-1))
        # print(len(decoder_state))
        # print(logits.shape)
        # print(decoder_state[-1].shape)
        

feature = torch.cat(Feature, 0)
score = torch.cat(Score_distribution, 0)
tokens = torch.cat(tokens, 0)
print(feature.shape)
print(score.shape)
print(tokens.shape)



