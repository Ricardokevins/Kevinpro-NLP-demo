from trainer import CharDataset
from trainer import TrainerConfig
from trainer import TransformerTrainer
from model import Transformer
model = Transformer()
train_dataset = CharDataset()
tconf = TrainerConfig(max_epochs=4 , batch_size=32, learning_rate=6e-4, lr_decay=True, num_workers=0)
trainer = TransformerTrainer(model, train_dataset, None, tconf)
trainer.train() 

