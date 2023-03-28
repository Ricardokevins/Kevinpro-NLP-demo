import pandas as pd     
import codecs
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer


if __name__ == "__main__":
    
    #文档预料 空格连接
    corpus = []
    

    import re


    path = "/Users/sheshuaijie/Desktop/RearchSpace/Data/Data/SAMSum/test.json"
    import json
    f = open(path,'r')
    data = json.load(f)
    f.close()

    import string
    punctuation_string = string.punctuation + '’'
    def re_punctuation(stri):
        for i in punctuation_string:
            stri = stri.replace(i, ' ').lower() 
        return stri

    def return_biturn(turns):
        biturns = []
        for i in range(0,len(turns)-1):
            biturns.append(turns[i]+turns[i+1])
        
        return biturns

    Document_Features = []
    for i in range(len(data)):
        dialogue = data[i]['dialogue']
        turns = dialogue.split('\n')
        turns = [i.strip()[len(i.split(":")[0])+2:] for i in turns]
        turns = [re_punctuation(i)for i in turns]
        #print(turns)
        turns = list(set(turns))
        # print(turns)
        # exit()
        turns = [i.split(" ") for i in turns]
        turns = [[j for j in i if len(j)!=0] for i in turns]
        turns = [" ".join(i) for i in turns]
        
        all_tokens = []
        #biturn = return_biturn(turns)
        #Document_Features.extend(biturn)


        corpus.extend(turns)

    
    #将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = CountVectorizer(min_df=5)
 
    #该类会统计每个词语的tf-idf权值
    transformer = TfidfTransformer()
 
    #第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

    #获取词袋模型中的所有词语  
    word = vectorizer.get_feature_names()
    
    #将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
    weight = tfidf.toarray()

 
    #打印特征向量文本内容
    # resName = "Tfidf_Result.txt"
    # result = codecs.open(resName, 'w', 'utf-8')
    # for j in range(len(word)):
    #     result.write(word[j] + ' ')
    # result.write('\r\n\r\n')
 
    # #每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重  
    # for i in range(len(weight)):
    #     for j in range(len(word)):
    #         result.write(str(weight[i][j]) + ' ')
    #     result.write('\r\n\r\n')
    # result.close()

    print( 'Start Kmeans:')
    from sklearn.cluster import KMeans
    clf = KMeans(n_clusters=100)   #科技 医学 汽车 国家
    s = clf.fit(weight)
    print( 'Start Kmeans:')
    #每个样本所属的簇
    label = []               
    i = 1
    while i <= len(clf.labels_):
        label.append(clf.labels_[i-1])
        i = i + 1
        
    y_pred = clf.labels_

    label2text = {}
    for i,j in zip(y_pred,corpus):
        if i in label2text:
            label2text[i].append(j)
        else:
            label2text[i] = [j]
    
    for i in label2text:
        f = open("cluster/"+str(i)+".txt",'w')
        for j in label2text[i]:
            f.write(j+"\n")
        f.close()
        
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=2)             #输出两维
    # newData = pca.fit_transform(weight)   #载入N维
    
    # xs, ys = newData[:, 0], newData[:, 1]
    # #设置颜色
    # cluster_colors = {0: 'r', 1: 'yellow', 2: 'b', 3: 'chartreuse', 4: 'purple', 5: '#FFC0CB', 6: '#6A5ACD', 7: '#98FB98'}
                  
    # #设置类名
    # cluster_names = {0: u'类0', 1: u'类1',2: u'类2',3: u'类3',4: u'类4',5: u'类5',6: u'类6',7: u'类7'} 
    
    # df = pd.DataFrame(dict(x=xs, y=ys, label=y_pred, title=corpus)) 
    # groups = df.groupby('label')
    
    # fig, ax = plt.subplots(figsize=(8, 5)) # set size
    # ax.margins(0.02)
    # for name, group in groups:
    #     ax.plot(group.x, group.y, marker='o', linestyle='', ms=10, label=cluster_names[name], color=cluster_colors[name], mec='none')
     
    # plt.show()