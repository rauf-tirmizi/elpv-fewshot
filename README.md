# elpv-fewshot

Code from my master's research project at ANU: few-shot crack detection on electroluminescence (EL) images of solar cells.

The question was simple. If a solar farm operator can only label a handful of cracked cells, can a prototypical network beat
just fine-tuning a pretrained CNN on those same few labels?

On the public ELPV dataset (2,624 cells, 300x300 greyscale) with 20 labelled examples per class:

| method | macro F1 |
|---|---|
| fine-tuned ResNet-18 | 0.61 |
| prototypical network, ResNet-18 encoder | 0.74 |

Averaged over 50 random support sets. Your numbers will move a bit with the seed.

## Data

Get the dataset from https://github.com/zae-bayern/elpv-dataset and point `--root` at the folder that has `labels.csv`
and `images/`. I treat defect probability 0 as "ok" and anything at or above 0.67 as "defect", and drop the uncertain
middle, which is roughly what the original paper does for the two-class setup.

## Running

```
pip install -r requirements.txt
python -m fewshot.train --root ~/data/elpv --shots 20 --episodes 2000
python -m fewshot.evaluate --root ~/data/elpv --ckpt runs/proto.pt --shots 20 --repeats 50
python -m fewshot.baseline --root ~/data/elpv --shots 20 --repeats 50
```

Training takes about 25 minutes on one RTX 3090.

## Notes

- The mono vs poly cell type matters more than I expected. Splitting by type and training separately did not help though.
- Augmentation is just flips and small rotations. Anything heavier smeared the thin cracks.
- This is research code, tidied enough for someone else to run it, not much more.
