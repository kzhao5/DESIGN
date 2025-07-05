import torch

from train import train_gcn
from utils import get_dataset

datasets = ["Yelp", "proteins", "Karate", "Cora", "CiteSeer", "PubMed"]

# get data
for d in datasets:
    dataset, data = get_dataset(d)
    if d == "proteins" or d == "Yelp":
        model = train_gcn(data, len(data.y[1]), data.train_mask, data.val_mask, data.test_mask, criterion=torch.nn.BCEWithLogitsLoss())
    else:
        model = train_gcn(data, dataset.num_classes, data.train_mask, data.val_mask, data.test_mask)
    torch.save(model, f"./models/{d}.pt")
