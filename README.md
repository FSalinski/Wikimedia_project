# Wikimedia project

Indoor vs outdoor image classifier for the course "Selected Topics in Deep Learning".

## What we are building
- Create a candidate pool of 20k-30k images from WIT.
- Label 500 images manually for a small gold set.
- Use CLIP to pseudo-label the rest and fine-tune a ResNet model.

## Folders
- `pool/inside/` and `pool/outside/` for the candidate pool.
- `train_set/inside/` and `train_set/outside/` for the final training set.

## Run
Create the candidate pool:

```bash
python scripts/create_candidate_pool.py --output_dir pool --target_total 20000
```

Manual labeling app (gold set):

```bash
python scripts/label_app.py --input_dir pool --output_csv labels_gold.csv --limit 500
```

Split the gold set into train/val/test inside the pool:

```bash
python scripts/split_gold.py --input_csv labels_gold.csv --out_dir pool/splits --test_size 200 --val_size 160 --copy_images
```

Pseudo-label the remaining pool and create train/val sets:

```bash
python scripts/pseudo_label_pool.py --pool_dir pool --splits_dir pool/splits --train_dir train_set --val_dir val_set
```

## Notebook
- [notebooks/main.ipynb](notebooks/main.ipynb)
