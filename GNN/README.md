GNN实现用于文本分类



暂时的实现是把每一个token作为节点

然后和当前节点的前三个节点之间用边相连

```python
def convert_sent2graph(sent):
    node = []
    edge = []
    step = [-3,-2,-1]
    for i in range(len(sent)):
        if sent[i]!=0:
            node.append(sent[i])
            for j in step:
                if i+j>=0 and i+j<len(sent) and sent[i+j]!=0:
                    edge.append([i,i+j])
                    edge.append([i+j,i])
    #node = range(2462)
    return node,edge
```



模型使用如下

```python
class GCN(torch.nn.Module):
    def __init__(self, num_node_features,num_classes,hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
 
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

```

