import sys
sys.path.append("..")

from trainer import CharDataset
from trainer import TrainerConfig
from trainer import TransformerTrainer
from trainer import TransformerDDPTrainer
from trainer import DataConfig
from model import make_model

import torch
import warnings
warnings.filterwarnings("ignore")
default_config = DataConfig()
default_config.max_samples = 2000
Train = True
#3.31 1.07
#3.22 0.58
# 0.52 0.19
if __name__=='__main__':
    if Train:
        model = make_model(7000,7000)
        train_dataset = CharDataset(default_config)
        tconf = TrainerConfig(max_epochs=1 , batch_size=8, learning_rate=2e-4, lr_decay=True, num_workers=20)
        trainer = TransformerDDPTrainer(model, train_dataset, None, tconf)
        #trainer = TransformerTrainer(model, train_dataset, None, tconf)
        trainer.train() 
    else:
        from trainer import DecodeData
        from trainer import greedy_decoder
        import warnings
        warnings.filterwarnings('ignore')

        model = make_model(7000,7000)
        model.load_state_dict(torch.load('./data/novelModel.pth'))

        #input = "头疼头晕怎么办"
        # 恶心想吐，不想吃饭咋办
        # 胃疼，反复拉肚子怎么办
        # 感觉自己有点发烧怎么办
        # 头疼，咳嗽，感觉有点发烧。我是不是得了新冠肺炎啊？
        # 每天睡不着，很晚睡，没有精神咋办
        # 我被玻璃划伤了脚，请问怎么处理
        question = "高血压患者能吃党参吗？,我有高血压这两天女婿来的时候给我拿了些党参泡水喝，您好高血压可以吃党参吗？"
        question = "恶心想吐，不想吃饭咋办"

        while 1 :
            question = input("输入你要问医生的问题      ：")
            result = greedy_decoder(model, input = question,config = default_config)
            print(result)
            print('\n')