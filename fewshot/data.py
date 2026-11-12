import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

DEFECT_THRESHOLD = 0.67


def load_labels(root):
    """labels.csv in the ELPV repo has no header: path, defect probability, cell type."""
    root = Path(root)
    df = pd.read_csv(root / "labels.csv", sep=r"\s+", header=None, names=["path", "prob", "kind"])
    df = df[(df.prob == 0) | (df.prob >= DEFECT_THRESHOLD)].copy()
    df["label"] = (df.prob >= DEFECT_THRESHOLD).astype(int)
    df["path"] = df.path.apply(lambda p: str(root / p))
    return df.reset_index(drop=True)


def make_transform(train):
    t = [transforms.Grayscale(num_output_channels=3), transforms.Resize(224)]
    if train:
        t += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(5)]
    t += [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return transforms.Compose(t)


class CellImages(Dataset):
    def __init__(self, df, train=False):
        self.df = df.reset_index(drop=True)
        self.tf = make_transform(train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row.path)
        return self.tf(img), int(row.label)


def split(df, test_frac=0.3, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_test = int(len(df) * test_frac)
    return df.iloc[idx[n_test:]], df.iloc[idx[:n_test]]


def sample_support(df, shots, seed):
    rng = random.Random(seed)
    parts = []
    for label in (0, 1):
        rows = df[df.label == label]
        parts.append(rows.iloc[rng.sample(range(len(rows)), shots)])
    return pd.concat(parts)


def episode(df, shots, queries, tf, seed):
    """One 2-way episode: (support x, support y, query x, query y)."""
    rng = random.Random(seed)
    sx, sy, qx, qy = [], [], [], []
    for label in (0, 1):
        rows = df[df.label == label]
        pick = rng.sample(range(len(rows)), shots + queries)
        for j, k in enumerate(pick):
            x = tf(Image.open(rows.iloc[k].path))
            if j < shots:
                sx.append(x); sy.append(label)
            else:
                qx.append(x); qy.append(label)
    return torch.stack(sx), torch.tensor(sy), torch.stack(qx), torch.tensor(qy)
