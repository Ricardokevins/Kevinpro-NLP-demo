import json
with open('/Users/sheshuaijie/Downloads/ANIML-master/ANIML.ipynb','r') as f:
    text=json.load(f)
# for i in text:
#     print(i)
#     print(text[i])
#     print('\n')
for x in text['cells']:
    if x['cell_type'] == 'code':
        for y in x.get('source'):
            print(y)