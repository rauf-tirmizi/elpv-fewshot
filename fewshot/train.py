import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import episode, load_labels, make_transform, split
from .model import ProtoNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--shots", type=int, default=20)
    ap.add_argument("--queries", type=int, default=15)
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="runs/proto.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_df, _ = split(load_labels(args.root))
    tf = make_transform(train=True)
    model = ProtoNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for ep in range(args.episodes):
        sx, sy, qx, qy = episode(train_df, args.shots, args.queries, tf, seed=ep)
        logits = model(sx.to(device), sy.to(device), qx.to(device))
        loss = F.cross_entropy(logits, qy.to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep % 100 == 0:
            acc = (logits.argmax(1).cpu() == qy).float().mean().item()
            print(f"episode {ep:5d}  loss {loss.item():.3f}  acc {acc:.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)


if __name__ == "__main__":
    main()
