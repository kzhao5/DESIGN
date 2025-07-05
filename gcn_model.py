import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# simple 2 layer GCN
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, bias=False)
        self.conv2 = GCNConv(hidden_channels, out_channels, bias=False)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x**2  # polynomial activation, for FHE
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
