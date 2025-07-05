import time
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics import f1_score, roc_auc_score

from gcn_model import GCN

# 1 epoch
def train(model: GCN, data: Data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()
    return loss.item()

# test function
def test(model: GCN, data: Data, train_mask, val_mask, test_mask):
    model.eval()

    start_time = time.time()
    out = model(data.x, data.edge_index)
    total_time = (time.time() - start_time) * 1000

    logits = F.log_softmax(out, dim=1)

    if len(data.y.shape) == 1:
        preds = logits.argmax(dim=1)
    else:
        # multilabel classification
        probs = torch.sigmoid(out)        # Convert logits to probabilities
        preds = (probs >= 0.5).int()      # Threshold to get 0's and 1's

    accs = []
    for mask in [train_mask, val_mask, test_mask]:
        if len(data.y.shape) == 1:
            correct = (preds[mask] == data.y[mask]).sum()
            acc = int(correct) / int(mask.sum())
        else:
            # multilabel classification
            acc = roc_auc_score(data.y[mask].reshape(-1), preds[mask].reshape(-1), average='micro')
        accs.append(acc)

    return accs[0], accs[1], accs[2], total_time

def train_gcn(data: Data, num_classes: int, train_mask, val_mask, test_mask, epochs: int = 100, hidden_dim: int = 16, lr: float = 0.01, weight_decay: float = 5e-4, dropout: float = 0.5, criterion = torch.nn.CrossEntropyLoss()):
    # get model
    model = GCN(data.num_node_features, hidden_dim, num_classes, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # train
    for i in range(epochs):
        loss = train(model, data, optimizer, criterion)
        train_acc, val_acc, test_acc, inference_time = test(model, data, train_mask, val_mask, test_mask)

        print(f"epoch {i} complete. loss: {loss:.4f}, train acc: {train_acc:.4f}, val acc: {val_acc:.4f}, test acc: {test_acc:.4f}, inference time (ms): {inference_time:.4f}")
        # print(f"epoch {i} complete. loss: {loss:.4f}")

    return model
