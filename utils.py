import torch
import random
import numpy as np
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph, scatter, subgraph
from torch_geometric.data import Data
from torch_geometric.datasets import KarateClub, Planetoid, Yelp
from torch_geometric.transforms import RandomNodeSplit, ToUndirected
from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.metrics import f1_score, roc_auc_score


def get_A_bar(edge_index, num_nodes):
    # A_norm = D^(-1/2) (A+I) D^(-1/2)
    #
    # edge_index = [2, num_edges]; edge_index[0][i] = source, edge_index[1][i] = target

    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float)

    # get edges from edge_index
    for i in range(edge_index.size(1)):
        src = edge_index[0][i]
        dst = edge_index[1][i]
        A[src, dst] = 1.0

    A += torch.eye(num_nodes)   # add self loops

    # get diagonal degree matrix, D
    deg = A.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    D_bar = torch.diag(deg_inv_sqrt)

    return D_bar @ A @ D_bar

def get_A_unnormalized(edge_index, num_nodes, add_self_loops=True):
    """
    Returns the unnormalized adjacency matrix A or A+I.
    """
    A = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0].float()
    if add_self_loops:
        A += torch.eye(num_nodes)
    return A


def get_subgraph(data):
    # get subgraph
    subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(node_idx=data.val_mask+data.test_mask+data.train_mask, edge_index=data.edge_index, num_hops=1, relabel_nodes=True)
    subdata = Data(x=data.x[subset], edge_index=sub_edge_index, y=data.y[subset])

    # get subgraph train, val, test masks
    new_train_mask = data.train_mask[subset]
    new_val_mask = data.val_mask[subset]
    new_test_mask = data.test_mask[subset]

    subdata.train_mask = new_train_mask
    subdata.val_mask = new_val_mask
    subdata.test_mask = new_test_mask

    return subdata

def get_predictions(output, multilabel=False):
    logits = F.log_softmax(output, dim=1)
    if not multilabel:
        preds = logits.argmax(dim=1)
    else:
        probs = torch.sigmoid(output)     # Convert logits to probabilities
        preds = (probs >= 0.5).int()      # Threshold to get 0's and 1's
    return preds

def test_acc_calc(preds, data):
    if len(data.y.shape) == 1:
        correct = (preds == data.y).sum()
        test_acc = int(correct) / int(data.num_nodes)
    else:
        test_acc = roc_auc_score(data.y.reshape(-1), preds.reshape(-1), average='micro')
    return test_acc


def get_dataset(name):
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    if name == "Karate":
        dataset = KarateClub(transform=RandomNodeSplit(num_train_per_class=None, num_test=0.2, num_val=0.15))
        data = dataset[0]
    elif name in ["Cora", "CiteSeer", "PubMed"]:
        dataset = Planetoid('./data', name=name)
        data = dataset[0]
        # if name == "PubMed":
        #     data = get_subgraph(data)
    elif name == "Yelp":
        dataset = Yelp(root='./data')
        data = dataset[0]
    elif name == "proteins":
        dataset = PygNodePropPredDataset(name="ogbn-proteins", root="./data")
        data = dataset[0]

        data.node_species = None
        data.y = data.y.to(torch.float)

        # Initialize features of nodes by aggregating edge features.
        row, col = data.edge_index
        data.x = scatter(data.edge_attr, col, dim_size=data.num_nodes, reduce='sum')

    if name == "proteins" or name == "Yelp" or name == "PubMed":
        random_nodes = torch.randperm(data.num_nodes)[:4000]

        subgraph_edge_index, subgraph_edge_attr = subgraph(random_nodes, data.edge_index, edge_attr=data.edge_attr, num_nodes=data.num_nodes, relabel_nodes=True)

        data = Data(x=data.x[random_nodes], edge_index=subgraph_edge_index, y=data.y[random_nodes], edge_attr=subgraph_edge_attr)
        t = RandomNodeSplit(num_test=0.2, num_val=0.2)
        data = t(data)

    print(f"------------- num nodes = {data.num_nodes} -------------")

    return dataset, data
