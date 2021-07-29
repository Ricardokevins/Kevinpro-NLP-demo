
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch
from torch.nn import Linear

import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, num_node_features,num_classes,hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, hidden_channels*2)
        self.conv2 = GCNConv(hidden_channels*2, hidden_channels*2)
        self.conv3 = GCNConv(hidden_channels*2, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin(x)
        
        return x
