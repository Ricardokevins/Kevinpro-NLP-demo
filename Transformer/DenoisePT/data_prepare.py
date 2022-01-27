from EasyTransformer import file_utils as fop

def preprocess():
    lines = fop.read_txt_lines('./data/cnews.train.txt')
    lines = [i.split('\t')[1] for i in lines]
    print(lines[0])
    fop.write_txt_file('./data/corpus.txt', lines)

preprocess()
