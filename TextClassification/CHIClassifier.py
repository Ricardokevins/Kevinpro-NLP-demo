def load_stop_word():
    f = open('cn_stopwords.txt',encoding='utf-8')
    words = f.readlines()
    words = [i.strip() for i in words]
    return words

def get_dataset(corpur_path):
    texts = []
    labels = []
    with open(corpur_path) as f:
        li = []
        while True:
            content = f.readline().replace('\n', '')
            if not content:              #为空行，表示取完一次数据（一次的数据保存在li中）
                if not li:               #如果列表也为空，则表示数据读完，结束循环
                    break
                label = li[0][10]
                text = li[1][6:-7]
                texts.append(text)
                labels.append(int(label))
                li = []
            else:
                li.append(content)       #["<Polarity>标签</Polarity>", "<text>句子内容</text>"]
    return texts, labels

import os
def makedir(path):
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    
def data_preprocess():
    text,label = get_dataset('trains.txt')
    import os
   
    makedir('sample')
    makedir('sample/0') 
    makedir('sample/1')
    i = 0
    for t,l in zip(text,label):
        path = 'sample/{}/{}.txt'.format(l,i)
        f = open(path,'w')
        f.write(t)
        i += 1


import os

from tqdm import tqdm
TopK_Features = 150
def load_data():
    #stop_list = load_stop_word()
    stop_list = []
    folder_list = os.listdir('sample')
    label2id = {}
    id2label = {}
    labels_samples = []
    for index,name in enumerate(folder_list):
        id2label[index] = name
        label2id[name] = index
    print(label2id)
    all_documents = []
    all_labels = []

    A_Matrix = []

    for label in label2id:
        Document_Matrix = {}

        file_dir = 'sample/{}/'.format(label)
        file_list = os.listdir(file_dir)
        labels_samples.append(len(file_list))
        for j in file_list:
            file_path = file_dir + '/{}'.format(j)
            f = open(file_path,encoding='utf-8')
            lines = f.readlines()
            lines = [i.strip() for i in lines]
            lines = " ".join(lines)
            
            tokens = []
            for i in lines.split(' '):
                if i not in stop_list:
                    tokens.append(i)
            all_documents.append(tokens)
            all_labels.append(label2id[label])
            token_set = list(set(tokens))
            for t in token_set:
                if t in Document_Matrix:
                    Document_Matrix[t] += 1
                else:
                    Document_Matrix[t] = 1
            


        A_Matrix.append(Document_Matrix)

    B_Matrix = []
    for index,Matrix in enumerate(A_Matrix):
        Document_Matrix = {}
        for word in Matrix:
            Document_Matrix[word] = 0
            for temp in range(len(A_Matrix)):
                if temp!=index and word in A_Matrix[temp]:
                    Document_Matrix[word]+=A_Matrix[temp][word]
        B_Matrix.append(Document_Matrix)
    '''
    temp = (A[i][word] * (M - B[i][word]) - (count[i] - A[i][word]) * B[i][word]) ^ 2 / (
            (A[i][word] + B[i][word]) * (N - A[i][word] - B[i][word]))
    '''
    Features = []
    IDFS = []
    for category in range(len(label2id)):
        CHI_Val = {}
        B_ADD_D = sum(labels_samples)-labels_samples[category]
        A_ADD_C = labels_samples[category]
        for token in A_Matrix[category]:    
            D = B_ADD_D - B_Matrix[category][token]
            C = A_ADD_C - A_Matrix[category][token]
            A = A_Matrix[category][token]
            B = B_Matrix[category][token]
            CHI_Val[token] = (A*D - B*C)**2*sum(labels_samples)
            under = (A+C)*(A+B)*(B+D)*(C+D)+0.00001
            CHI_Val[token] = CHI_Val[token]/under
        Feature_word = sorted(CHI_Val.items(), key=lambda t: t[1], reverse=True)[:TopK_Features]    
        Feature_word = [i[0] for i in Feature_word]  
        
        IDF_VAL = {}
        #N/(A[cla][word]+B[cla][word])
        for token in Feature_word:
            D = B_ADD_D - B_Matrix[category][token]
            C = A_ADD_C - A_Matrix[category][token]
            A = A_Matrix[category][token]
            B = B_Matrix[category][token]
            idf = (A+B+C+D)/(A+B)
            IDF_VAL[token] = idf
        IDFS.append(IDF_VAL)
        Features.append(Feature_word)
    # print('nba==================================')
    # print(A_Matrix[0]['NBA'])
    # print(B_Matrix[0]['NBA'])
    return all_documents,all_labels,Features,IDFS


do,la,features,IDFS = load_data()

import math
def extract(do,la,features,IDFS):
    TF = {}
    for i in do:
        #print(i)
        if i in TF:
            TF[i] += 1
        else:
            TF[i] = 1
    for i in TF:
        TF[i] = TF[i]/len(do)

    import numpy
    All_features = []
    for index,feature_word in enumerate(features):
        Feature_Array = []
        for j in feature_word:
            if j not in TF:
                Feature_Array.append(0)
            else:
                #Feature_Array.append(1)
                #Feature_Array.append(TF[j])
                Feature_Array.append(TF[j]*math.log(IDFS[index][j]))
        All_features.append(Feature_Array)

    Final_features = numpy.array(All_features)
    #print(Final_features.shape)
    return Final_features.reshape(-1)

train_sample_num = 900
test_sample_num = 100

train_index = []
test_index = []

import random
import random
random.seed(1) 
import numpy as np
np.random.seed(1)

while len(train_index)<train_sample_num:
    index = random.randint(0,len(do)-1)
    if index not in train_index:
        train_index.append(index)
for i in range(len(do)):
    if i not in train_index:
        test_index.append(i)
print("trains samples {}".format(len(train_index)))
print("tests samples {}".format(len(test_index)))

train_samples = []
train_labels = []
test_samples = []
test_labels = []


import random
for index in train_index:
    train_samples.append(extract(do[index],la[index],features,IDFS))
    train_labels.append(la[index])

for index in test_index:
    test_samples.append(extract(do[index],la[index],features,IDFS))
    test_labels.append(la[index])

from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
svc = svm.SVC()  # 支持向量机，SVM
mlp = MLPClassifier()
model = mlp
#model = svc
model = LogisticRegression()
model.fit(train_samples, train_labels)
print('begin to predict……')
y_pred_model = model.predict(test_samples)
result = [la[index] for index in test_index]
right =0 
for i,j in zip(y_pred_model,result):
    if i==j:
        right += 1
print("Total {} Right {} Acc Ratio {}%".format(len(test_index),right,right/len(test_index)*100))





"""
        OneHot     BOW     TFIDF
MLP     73.0%     78.0%    77.0%
SVM     66.0%     70.0%    75.0%
LR      72.0%     70.0%    73.0%
"""