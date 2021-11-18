import jieba
from jieba import analyse

datas = []
Features = []

f = open("data1.txt",encoding="utf-8")

data1 = f.read()
feature1 = jieba.analyse.extract_tags(data1, topK=30, withWeight=False, allowPOS=())
Features.append(feature1)
datas.append(data1)

f = open("data2.txt",encoding="utf-8")
data2 = f.read()
feature2 = jieba.analyse.extract_tags(data2, topK=30, withWeight=False, allowPOS=())
Features.append(feature2)
datas.append(data2)

f = open("data3.txt",encoding="utf-8")
data3 = f.read()
datas.append(data3)
feature3 = jieba.analyse.extract_tags(data3, topK=30, withWeight=False, allowPOS=())
Features.append(feature3)



query = '美国是啥'
query_feature = jieba.analyse.extract_tags(query, topK=3, withWeight=False, allowPOS=())
print(query_feature)
max_match = 0
target = 0
for index in range(len(Features)):
    score = 0
    for j in query_feature:
        for t in Features[index]:
            if j == t:
                score+=1
    if score>max_match:
        max_match=score
        target = index
print(datas[target])


