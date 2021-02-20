from dataloader import SumDataset
import torch
from torch.utils.data import Dataset, DataLoader

from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import FileStorageObserver

ex = Experiment('PGNSum')
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(FileStorageObserver('Train'))
# obv = MongoObserver(url="localhost", port=27017, db_name="PgnSum")
# ex.observers.append(obv)

@ex.config
def config():
    batch_size = 2
    epochs = 5
    max_len = 128
    cuda_device=[0]
    optimizer={'choose':'AdamW','lr':2e-5,'eps':1e-8}
    loss_func='CE'
    possessbar = False
    


class Trainer:
    def __init__(self,config,runner):
        self.runner = runner
        self.use_possessbar=config['possessbar']
        self.batch_size = config["batch_size"]
        self.epoch = config['epochs']
        self.device_ids=config["cuda_device"]
        self.model=BERT_MOM(int(self.batch_size/len(self.device_ids)))
        if config['optimizer']['choose'] == "AdamW":
            self.optim = AdamW(self.model.parameters(),lr = config['optimizer']['lr'], eps = config['optimizer']['eps'])
        elif config['optimizer']['choose'] == "SGD":
            self.optim = optim.SGD(self.model.parameters(), lr=config['optimizer']['lr'])
        else:
            print("Hit error")
            exit()
        
        if config['loss_func']=="CE":
            self.criterion = nn.CrossEntropyLoss()
        
        data = SumDataset()
        self.dataloader=DataLoader(data, batch_size=self.batch_size, shuffle=True)

        #self.model=nn.DataParallel(self.model,device_ids=self.device_ids)
        self.model=self.model.cuda()

@ex.main
def main(_config,_run):
    print("----Runing----")
    trainer = Trainer(_config,_run)
    trainer.train()


if __name__ == "__main__":
    r=ex.run()
    print(r.config)
    print(r.host_info)