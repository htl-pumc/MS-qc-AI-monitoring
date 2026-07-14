# Trained models

The seven trained checkpoints are distributed through the companion data/model archive:

<https://doi.org/10.5281/zenodo.21337993>

Expected files:

```text
fusion2.pt
fusion3.pt
lumos1.pt
qe_hf3.pt
qe_hf4.pt
qe_plus.pt
m3.pt
```

Each checkpoint is a single self-describing PyTorch dictionary containing:

- instrument identifier;
- DDA or DIA preprocessing profile;
- exact feature names and order;
- VAE and MLP dimensions;
- trained state dictionaries;
- classification threshold;
- training hyperparameters.

Use `scripts/run_model.py predict` to validate the input feature schema and apply a checkpoint.
