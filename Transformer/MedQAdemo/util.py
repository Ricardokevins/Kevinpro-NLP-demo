def readFromPair():
    f = open('./data/question.txt','r',encoding='utf-8')
    question = f.readlines()
    question = [i.replace('\n','') for i in question]
    f = open("./data/answer.txt",'r',encoding = 'utf-8')
    answer = f.readlines()
    answer = [i.replace('\n','') for i in answer]

    return question,answer
