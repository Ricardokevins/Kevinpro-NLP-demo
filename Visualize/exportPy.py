import json
with open('filename.ipynb','r') as f:
    text=json.load(f)
for x in text[keyss[-1]]:
    for y in x.get('source'):
        print(y)