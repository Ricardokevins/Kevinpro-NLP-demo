
import csv

csvFile = open("data/question.csv", "r",encoding='utf-8')
reader = csv.reader(csvFile)
question = {}
head = next(reader)
for item in reader:
    question[item[0]] = item[1]

print(len(question))

csvFile = open("data/answer.csv", "r",encoding='utf-8')
reader = csv.reader(csvFile)
question = {}
head = next(reader)
print(head)
paired = 0
for item in reader:
    if item[1] in question:
        print(item)
        paired += 1
        
    else:
        print("hit unpaired")
        continue


print(len(question))
