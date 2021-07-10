import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import DataLoader
from tqdm import tqdm
from torch_geometric.data import Data
import transformer
from dataload import TrainDataset
train_dataset = TrainDataset(root = './')
from dataload import TestDataset
test_dataset = TestDataset(root = './')
#test_dataset = TrainDataset(root = './')
print(len(train_dataset))
print(len(test_dataset))

from model import GCN



model = GCN(256,2,hidden_channels=64)
print(model)
  
device = torch.device('cuda')
#model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
criterion = torch.nn.CrossEntropyLoss()

batch_size = 4
num_epochs = 10
train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=2)
model.train()

from tqdm import tqdm

def test():
    model.eval()
    
    correct = 0
    for data in test_loader:  # Iterate in batches over the training/test dataset.
        #data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(test_loader.dataset)  # Derive ratio of correct predictions.

def train():
    for i in range(num_epochs):
        model.train()
        for data in tqdm(train_loader):  # Iterate in batches over the training dataset.
            #data = data.to(device)
            #print(data.x.shape)
            out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            loss = criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
        acc = test()
        print("Epoch {} Acc {}".format(i,acc))

train()