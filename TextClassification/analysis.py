import torch

model_dict = torch.load('model1.pth')
model_dict.eval()
print(model_dict)

#sample = 'great bagels made the old-fashioned way.'
sample = 'we visited bread bar during january restaurant week and were so pleased with the menu selections and service.'
sample = 'disappointing food, lousy service.'

from dataloader import build_tokenizer
import matplotlib.pyplot as plt
from matplotlib import ticker
tokenzier = build_tokenizer()

encode = tokenzier.encode(sample)
model_dict=model_dict.cuda()
sents = []
sents.append(encode)
sent = torch.tensor(sents)
print(sent.shape)
sent = sent.cuda()
logits, attn = model_dict(sent)
print(logits)
import seaborn as sns 
def display_attention(candidate,attention):
    fig=plt.figure(figsize=(10,10))
    ax=fig.add_subplot(111)
    attention = attention.squeeze(1).cpu().detach().numpy()
    fontdict = {'rotation': 90} 
    cax = ax.matshow(attention,cmap="RdBu_r")
    fig.colorbar(cax)
    ax.tick_params(labelsize=15)
    ax.set_xticklabels(['']+candidate,fontdict=fontdict)
    ax.set_yticklabels(['']+candidate)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    #plt.close()

def display_attention_(candidate, attention):
    
    attention=attention.squeeze(1).cpu().detach().numpy()
    ax = sns.heatmap(attention,cmap="RdBu_r")

    ax.set_xticklabels(candidate)
    ax.set_yticklabels(candidate)
    label_x = ax.get_xticklabels()
    plt.setp(label_x, rotation=90)
    
    label_y = ax.get_yticklabels()
    plt.setp(label_y , rotation = 0)
    plt.show()

show_length = 18
attn = attn.reshape(64, 64)
attn = attn[:show_length,:show_length]
display_attention(sample.split(' ')[:show_length],attn)
display_attention_(sample.split(' ')[:show_length],attn)
# 
# attn = attn.reshape(-1, 64)
# attn = attn[:, :show_length]
# print(attn)
# display_attention(sample.split(' ')[:show_length], attn)
# display_attention_(sample.split(' ')[:show_length], attn)

