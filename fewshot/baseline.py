"""Fine-tune a pretrained ResNet-18 on the same few labelled cells, for comparison."""
import argparse

import numpy as np
import timm
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from .data import CellImages, load_labels, sample_support, split


def run_once(train_df, test_df, shots, seed, device, epochs=30):
    torch.manual_seed(seed)
    support = sample_support(train_df, shots, seed=1000 + seed)
    model = timm.create_model("resnet18", pretrained=True, num_classes=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loader = DataLoader(CellImages(support, train=True), batch_size=16, shuffle=True)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            loss = F.cross_entropy(model(x.to(device)), y.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for x, y in DataLoader(CellImages(test_df), batch_size=64):
            preds.append(model(x.to(device)).argmax(1).cpu()); ys.append(y)
    return f1_score(torch.cat(ys), torch.cat(preds), average="macro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--shots", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=50)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_df, test_df = split(load_labels(args.root))
    scores = [run_once(train_df, test_df, args.shots, r, device) for r in range(args.repeats)]
    print(f"baseline macro F1: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")


if __name__ == "__main__":
    main()
