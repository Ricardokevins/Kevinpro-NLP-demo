

from parrot import Parrot
import torch
import warnings
warnings.filterwarnings("ignore")

''' 
uncomment to get reproducable paraphrase generations
def random_state(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

random_state(1234)
'''

#Init models (make sure you init ONLY once if you integrate this to your code)
parrot = Parrot(model_tag="/Users/sheshuaijie/Downloads/parrot_paraphraser_on_T5")

phrases = ["Can you recommend some upscale restaurants in Newyork?",
           "What are the famous places we should not miss in Russia?"
]

# for phrase in phrases:
#   print("-"*100)
#   print("Input_phrase: ", phrase)
#   print("-"*100)
#   para_phrases = parrot.augment(input_phrase=phrase, use_gpu=False,max_return_phrases = 10, 
#                                max_length=128)
#   for para_phrase in para_phrases:
#    print(para_phrase)

from tqdm import tqdm
def paraphrasing_files():
    f = open("all_target.txt",'r')
    with open("all_target_paraphrased.txt",'w') as f_dump:
        lines = f.readlines()
        results = []
        for i in tqdm(lines):
            para_phrases = parrot.augment(input_phrase=i, use_gpu=False,max_return_phrases = 10, max_length=128)
            if para_phrases:
                result = [j[0] for j in para_phrases]
                result = "(%$%$%$)".join(result)
                f_dump.write(result+'\n')
            else:
                f_dump.write("NO PARAPHRASED FOUND" + '\n')

paraphrasing_files()