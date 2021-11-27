# import nltk

# tagged = nltk.corpus.brown.tagged_sents()[0]
# entity = nltk.chunk.ne_chunk(tagged)
# print (entity)
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_lg
import en_core_web_sm
import json
from tqdm import tqdm
import random
random.seed(413)

nlp = en_core_web_lg.load()
f = open('train.source','r',encoding = 'utf-8')
fout = open('train.extract.source','w',encoding = 'utf-8')
exclude = ['DATE','TIME','PERCENT','MONEY','QUANTITY','ORDINAL','CARDINAL']

lines = f.readlines()

for i in tqdm(range(len(lines))):
    data = lines[i].strip()
    doc= nlp(data)
    data_dict = {}
    extract_features = []
    for X in doc.ents:
        if X.label_ not in exclude:
            #print(X.label_,X.text)
            extract_features.append(X.text)
   # exit()
    data_dict['feature'] = extract_features
    data_dict['text'] = [str(i) for i in list(doc.sents)]

    #print(data_dict)
    fout.write(json.dumps(data_dict) + '\n')
# print([(X.text, X.label_)for X in doc.ents])

# # PERSON ORG GPE 
# print(doc.ents)
# print(type(doc.ents))
# for i in doc.sents:
#     print(i)
# print(list(doc.sents))

'''
PERSON:      People, including fictional.
NORP:        Nationalities or religious or political groups.
FAC:         Buildings, airports, highways, bridges, etc.
ORG:         Companies, agencies, institutions, etc.
GPE:         Countries, cities, states.
LOC:         Non-GPE locations, mountain ranges, bodies of water.
PRODUCT:     Objects, vehicles, foods, etc. (Not services.)
EVENT:       Named hurricanes, battles, wars, sports events, etc.
WORK_OF_ART: Titles of books, songs, etc.
LAW:         Named documents made into laws.
LANGUAGE:    Any named language.
DATE:        Absolute or relative dates or periods.
TIME:        Times smaller than a day.
PERCENT:     Percentage, including ”%“.
MONEY:       Monetary values, including unit.
QUANTITY:    Measurements, as of weight or distance.
ORDINAL:     “first”, “second”, etc.
CARDINAL:    Numerals that do not fall under another type.
'''