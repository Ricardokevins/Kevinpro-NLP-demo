import EasyTransformer.file_utils as fop

def utteranceIsQ(sent):
    if sent[0] == "回":
        return 0
    else:
        return 1

def merge_dialogue(utters):
    Q = []
    A = []
    flag = -1
    cache_sents = ""
    i = 0
    while i  < len(utters):
        if flag == -1:
            if utteranceIsQ(utters[i]) == 0:
                flag = 1
        else:
            if utteranceIsQ(utters[i]) == flag:
                cache_sents += utters[i][5:]
            else:
                if flag:
                    Q.append(cache_sents)
                else:
                    A.append(cache_sents)
                
                cache_sents = utters[i][5:]
                flag = utteranceIsQ(utters[i])
        i += 1
    if flag:
        Q.append(cache_sents)
    else:
        A.append(cache_sents)
    #你打印后下车，然后让学校去签字盖章
    assert len(A) <= len(Q)
    max_size = min(len(A), len(Q))
    return Q[:max_size],A[:max_size]

def process_data():
    data = fop.read_xlsx_file("D:/Python_plus/data_set/400.xlsx")
    sources = []
    targets = []
    max_length = 0
    for i in data:
        i = data[3]
        dialogue = i['通话内容']
        print(dialogue)
        utters = dialogue.strip().split('\r\n')
        #if len("".join(utters).replace("回答人: ", "").replace("咨询人: ", "").replace("\n","")) > max_length:
        Q,A = merge_dialogue(utters)
        print(Q)
        print(A)
        exit()
        #max_length += len("".join(utters).replace("回答人: ", "").replace("咨询人: ", "").replace("\n",""))
        # exit()
        # for j in range(len(utters)):
        #     pass
        #     # if utters[]
    print(max_length/len(data))
    print(data[0])

process_data()