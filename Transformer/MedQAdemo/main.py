from trainer import CharDataset
from trainer import TrainerConfig
from trainer import TransformerTrainer
from model import make_model

import torch

model = make_model(7000,7000)
train_dataset = CharDataset()
tconf = TrainerConfig(max_epochs=2 , batch_size=32, learning_rate=2e-4, lr_decay=True, num_workers=0)
trainer = TransformerTrainer(model, train_dataset, None, tconf)
trainer.train() 

