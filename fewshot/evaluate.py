import argparse

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from .data import CellImages, load_labels, sample_support, split
from .model import ProtoNet


@torch.no_grad()
def embed_all(model, df, device):
    loader = DataLoader(CellImages(df), batch_size=64)
    zs, ys = [], []
    for x, y in loader:
        zs.append(model.embed(x.to(device)).cpu())
        ys.append(y)
    return torch.cat(zs), torch.cat(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shots", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=50)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_df, test_df = split(load_labels(args.root))
    model = ProtoNet(pretrained=False).to(device).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    zq, yq = embed_all(model, test_df, device)
    scores = []
    for r in range(args.repeats):
        support = sample_support(train_df, args.shots, seed=1000 + r)
        zs, ys = embed_all(model, support, device)
        protos = torch.stack([zs[ys == c].mean(0) for c in (0, 1)])
        pred = torch.cdist(zq, protos).argmin(1)
        scores.append(f1_score(yq, pred, average="macro"))
    print(f"macro F1 over {args.repeats} support sets: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")


if __name__ == "__main__":
    main()
