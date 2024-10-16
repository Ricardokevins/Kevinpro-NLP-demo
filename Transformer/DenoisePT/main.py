import sys
sys.path.append("..")

from trainer import CharDataset
from trainer import TrainerConfig
from trainer import TransformerTrainer
from trainer import DataConfig
from model import make_model

import torch
import warnings
warnings.filterwarnings("ignore")
default_config = DataConfig()

Train = False
if Train:
    train_dataset = CharDataset(default_config)
    model = make_model(6300,6300)
    tconf = TrainerConfig(max_epochs=4 , batch_size=64, learning_rate=2e-4, lr_decay=True, num_workers=0)
    trainer = TransformerTrainer(model, train_dataset, None, tconf)
    trainer.train() 
else:
    from trainer import DecodeData
    from trainer import greedy_decoder
    import warnings
    warnings.filterwarnings('ignore')

    model = make_model(6300,6300)
    model.load_state_dict(torch.load('./data/novelModel.pth'))

    #input = "头疼头晕怎么办"
    # 恶心想吐，不想吃饭咋办
    # 胃疼，反复拉肚子怎么办
    # 感觉自己有点发烧怎么办
    # 头疼，咳嗽，感觉有点发烧。我是不是得了新冠肺炎啊？
    # 每天睡不着，很晚睡，没有精神咋办
    # 我被玻璃划伤了脚，请问怎么处理
    # 我想咨一下关于咱们本科招生强机计划的一问题


    question = "你懂我在什么"
    question = "恶心想吐，不想吃饭咋办"

    while 1 :
        question = input("输入：")
        result = greedy_decoder(model, input = question,config = default_config)
        print("输出：" + result)
        print('\n')