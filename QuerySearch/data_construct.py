import random
random.seed(413)
from tqdm import tqdm
sub_folder = "./subData/"
def extract_data(prefix,number):
    source_path = "{}.source".format(prefix)
    target_path = '{}.target'.format(prefix)
    f1 = open(source_path,'r',encoding = 'utf-8')
    f2 = open(target_path,'r',encoding = 'utf-8')
    source_data = f1.readlines()
    target_data = f2.readlines()
    Curnumber = len(source_data)
    print(Curnumber)
    data_index = []
    while len(data_index) < number:
        random_index = random.randint(0,Curnumber-1)
        if random_index not in data_index:
            data_index.append(random_index)
    print(data_index[:10])

    f3 = open(sub_folder + source_path,'w',encoding = 'utf-8')
    f4 = open(sub_folder + target_path,'w',encoding = 'utf-8')
    for i in data_index:
        f3.write(source_data[i].strip() + '\n')
        f4.write(target_data[i].strip() + '\n')




    
extract_data('train',5000)
extract_data('val',1000)

