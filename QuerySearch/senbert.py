from sentence_transformers import SentenceTransformer
import os
path = "/Users/sheshuaijie/Desktop/RearchSpace/Data/Data/SAMSum/train.json"
import json
f = open(path,'r')
data = json.load(f)
f.close()
Document_Features = []
for i in range(len(data)):
    dialogue = data[i]['dialogue']
    turns = dialogue.split('\n')
    turns = [i.strip()[len(i.split(":")[0])+2:] for i in turns]
    Document_Features.extend(turns)

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# dense_feature = []
# from tqdm import tqdm
# for sents in tqdm(Document_Features):
#     embed =  model.encode(sents)
#     dense_feature.append(embed)

# feature = torch.cat(Feature, 0)
import torch
feature = model.encode(Document_Features)
torch.save(feature, "featurefull2.pt")
print(feature.shape)
