PAD = '[pad]'
SENTENCE_START = '[s]'
SENTENCE_END = '[/s]'
OOV = '[oov]'

MIN_Word_count = 3
MAX_Sentence_length = 10
def read_file(filename):
    file = open(filename, 'r',encoding='utf8')
    lines = file.readlines()
    sent1s = []
    sent2s = []
    index = 0
    for i in lines:
        index += 1
        s1,s2 = i.split('\t')[0],i.split('\t')[1]

        s1 = s1.strip()
        s2 = s2.strip()

        if len(s1.split()) <= 2 or len(s2.split()) <= 2:
            continue
        if len(s1.split()) < MAX_Sentence_length and len(s2.split()) < MAX_Sentence_length:
            sent1s.append(s1)
            sent2s.append(s2)
    return sent1s,sent2s


class Tokenizer:
    def __init__(self, LoadFromFile=True):
        self.word2id = {}
        self.id2word = {}
        self.word2count = {}
        self.cur_word = 0
        for i in range(10):
            self.add_word(PAD)
            self.add_word(SENTENCE_START)
            self.add_word(SENTENCE_END)
            self.add_word(OOV)
        if LoadFromFile:
            self.load()

    def add_word(self, word):
        if word not in self.word2id:
            self.id2word[self.cur_word] = word
            self.word2id[word] = self.cur_word
            self.word2count[word] = 1
            self.cur_word += 1
        else:
            self.word2count[word]+=1

    def add_sentence(self, sentence):
        for i in sentence.split(' '):          
            self.add_word(i)


    def export(self):
        f = open('dict.txt', 'w', encoding='utf-8')
        for i in self.word2id:
            if self.word2count[i]>MIN_Word_count:
                f.write(i)
                f.write('\n')
        f.close()

    def load(self):
        f = open('dict.txt', 'r', encoding='utf-8')
        lines = f.readlines()
        for i in lines:
            word = i.strip()
            self.add_word(word)

# t = Tokenizer(False)
# sent1,sent2 = read_file("./formatted_movie_lines.txt")
# for i,j in zip(sent1,sent2):
#     t.add_sentence(i)
#     t.add_sentence(j)

# t.export()

# sent1,sent2 = read_file("./formatted_movie_lines.txt")
# f = open("target.txt",'w',encoding='utf-8')
# for i in sent2:
#     f.write(i +'\n')