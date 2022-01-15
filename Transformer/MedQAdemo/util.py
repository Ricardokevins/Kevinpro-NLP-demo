def readFromPair():
    f = open('./data/question.txt','r',encoding='utf-8')
    question = f.readlines()
    f = open("./data/answer.txt",'r',encoding = 'utf-8')
    answer = f.readlines()

    return question,answer
