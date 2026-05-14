# pip install torch torch-geometric networkx matplotlib

import torch
import torch.nn.functional as F
from torch_geometric.datasets import KarateClub
from torch_geometric.nn import GCNConv

dataset = KarateClub()
data = dataset[0]

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    model.train()
    optimizer.zero_grad()

    out = model(data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()

model.eval()
pred = model(data).argmax(dim=1)
acc = (pred == data.y).sum() / data.num_nodes

print("Accuracy:", acc.item())
print("Prediction:", pred.tolist())