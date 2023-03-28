import faiss
import torch
from sentence_transformers import SentenceTransformer
import json
import os
import time
path = "/Users/sheshuaijie/Desktop/RearchSpace/Data/Data/SAMSum/train.json"
f = open(path,'r')
data = json.load(f)
f.close()
Document_Features = []
for i in range(len(data)):
    dialogue = data[i]['dialogue']
    turns = dialogue.split('\n')
    turns = [i.strip()[len(i.split(":")[0])+2:] for i in turns]
    Document_Features.extend(turns)

feature = torch.load('featurefull2.pt')
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

d = 384
nlist = 1024
m = 64
k = 10



quantizer = faiss.IndexFlatL2(d)    # 内部的索引方式依然不变
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
if not os.path.exists("data.trained"):
    start = time.time()
    index.train(feature)
    print('Training took {} s'.format(time.time() - start))

    print('Writing index after training')
    start = time.time()
    faiss.write_index(index, "data.trained")
    print('Writing index took {} s'.format(time.time()-start))
index = faiss.read_index("data.trained")
# print('Adding Keys')
# index = faiss.read_index(args.faiss_index+".trained")
# start = args.starting_point
# start_time = time.time()
# while start < args.dstore_size:
#     end = min(args.dstore_size, start+args.num_keys_to_add_at_a_time)
#     to_add = keys[start:end].copy()
#     index.add_with_ids(to_add.astype(np.float32), np.arange(start, end))
#     start += args.num_keys_to_add_at_a_time

#     if (start % 1000000) == 0:
#         print('Added %d tokens so far' % start)
#         print('Writing Index', start)
#         faiss.write_index(index, args.faiss_index)

# print("Adding total %d keys" % start)
# print('Adding took {} s'.format(time.time() - start_time))
# print('Writing Index')
# start = time.time()
# faiss.write_index(index, args.faiss_index)
# print('Writing index took {} s'.format(time.time()-start))
# faiss.write_index(index, "data.trained")
# index.add(feature)


while 1:
    input_query = input("input query:")
    query = model.encode(input_query)
    query = query.reshape(1,384)
    D, I = index.search(query, k) 
    I = I.reshape(-1)
    I = I.tolist()
    for i in I:
        print(" ".join(Document_Features[i]))