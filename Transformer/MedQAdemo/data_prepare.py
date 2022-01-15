

asklist = []
answerlist = []
target_data = [
    "Andriatria_男科\男科5-13000.csv",
    "IM_内科\内科5000-33000.csv",
    "OAGD_妇产科\妇产科6-28000.csv",
    "Oncology_肿瘤科\肿瘤科5-10000.csv",
    "Pediatric_儿科\儿科5-14000.csv",
    "Surgical_外科\外科5-14000.csv"
]
for i in target_data:
    with open(i,encoding = 'utf-8') as f:
        for temp_lines in f.readlines()[1:]:
            lin = temp_lines[0:-1].split(',')
            if i==0:
                continue        
            #print(lin)
            if len(lin) == 4:
                if len(lin[1]+','+lin[2])<200 and len(lin[3])<200:
                    asklist.append(lin[1]+','+lin[2])
                    answerlist.append(lin[3])

print(len(asklist),len(answerlist))
question = open("question.txt",'w',encoding = 'utf-8')
answer = open("answer.txt",'w',encoding = 'utf-8')

for i in range(len(asklist)):
    question.write(asklist[i]+"\n")
    answer.write(answerlist[i] + "\n")
    