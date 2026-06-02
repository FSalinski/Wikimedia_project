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

## Notebook
- [notebooks/main.ipynb](notebooks/main.ipynb)
