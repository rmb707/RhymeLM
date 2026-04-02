# How to Use

## Install
```bash
pip install -e .
```

## Train a Model
```bash
# LSTM baseline
python train.py --config configs/lstm_base.yaml

# LSTM + Self-Attention hybrid
python train.py --config configs/lstm_attention.yaml

# Transformer (from scratch)
python train.py --config configs/transformer_char.yaml

# Quick test (fewer steps)
python train.py --config configs/lstm_base.yaml --steps 5000
```

## Generate Verses
```bash
# Basic generation
python generate.py --checkpoint checkpoints/rhymelm_lstm_final.pt --prompt "Yeah, "

# With rhyme scheme
python generate.py --checkpoint checkpoints/rhymelm_lstm_final.pt --rhyme-scheme AABB

# In an artist's style
python generate.py --checkpoint checkpoints/rhymelm_lstm_final.pt --artist "Eminem"

# With evaluation metrics
python generate.py --checkpoint checkpoints/rhymelm_lstm_final.pt --eval
```

## Notebooks
Interactive walkthroughs in `notebooks/`:
1. `01_train_lstm.ipynb` — training with the `rhymelm` package
2. `02_evaluate_and_visualize.ipynb` — embedding space, sampling strategies, metrics

## Original Notebook
The standalone `RhymeLM_v2.ipynb` still works independently — just run all cells top to bottom.
