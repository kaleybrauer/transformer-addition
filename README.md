# Transformer Addition
This repo trains a small causal Transformer from scratch to add two base-10 integers and evaluates generalization.

## Training
Configure the run in config/config.yaml (e.g., model size, epochs, batch size).

run:
```
python train.py
```

This writes checkpoints under:
```
exp/<hf_model_id>/<run_name>/checkpoint-XXXX/
```

## Evaluation

Edit MODEL_DIR in eval.py to point to a trained checkpoint, then run:
```
python eval.py
```

This produces results for each epoch in results folder.

Plot the results for different datasets with:
```
python plot.py
```
