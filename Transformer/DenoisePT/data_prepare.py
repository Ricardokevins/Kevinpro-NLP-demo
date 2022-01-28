from EasyTransformer import file_utils as fop
import random
from tqdm import tqdm
def get_temp_dict():
    lines = fop.read_txt_lines('./data/corpus.txt')
    temp_dict = set()
    for i in lines:
        for j in i:
            temp_dict.add(j)

    return list(temp_dict)

def apply_noise(sent,temp_dict):
    # Random delete
    # Random replace
    # Replace with similar spell?

    # 20% corrupt
    # 33% delete 
    # 33% insert
    # 33% replace
    after = []
    for i in range(len(sent)):
        noise_prob = random.random()
        if noise_prob <= 0.15:
            noise_prob = random.random()
            if noise_prob <= 0.40:
                continue
            if noise_prob <=0.60 and noise_prob > 0.40:
                random_char = random.choice(temp_dict)
                after.append(sent[i])
                after.append(random_char)
            if noise_prob > 0.60:
                random_char = random.choice(temp_dict)
                after.append(random_char)
        else:
            after.append(sent[i])
    return "".join(after)

def generate_noise_data():
    sents = fop.read_txt_lines('./data/corpus.txt')
    temp_dict = get_temp_dict()
    source = []
    target = []
    for e in range(10):
        for i in tqdm(sents):
            t = i[:200]
            s = apply_noise(t,temp_dict)
            source.append(s)
            target.append(t)
    fop.write_txt_file('./data/target.txt',target)
    fop.write_txt_file('./data/source.txt', source)


generate_noise_data()

def preprocess():
    lines = fop.read_txt_lines('./data/cnews.train.txt')
    lines = [i.split('\t')[1] for i in lines]
    print(lines[0])
    fop.write_txt_file('./data/corpus.txt', lines)


