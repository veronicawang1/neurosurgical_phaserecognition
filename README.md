# Neurosurgery Phase Recognition
AI model for phase recognition in aneurysm clipping using 1 FPS 1080p video.

## Pipeline

```bash
# 1. Labels from CSVs
python prepare_labels.py

# 2. (Optional) Pretrain frame CNN
python train_baseline.py --epochs 20

# 3. Extract features + frozen CNN logits (for OperA attention reg)
python extract_features.py --cnn_checkpoint checkpoints/baseline_cnn.pt

# Or add logits to existing feature files:
python precompute_cnn_logits.py --cnn_checkpoint checkpoints/baseline_cnn.pt

# 4. Train transformer with attention regularization
python train_transformer.py --lambda_reg 1.0 --cnn_checkpoint checkpoints/baseline_cnn.pt
```

Use `--lambda_reg 0` to disable attention regularization. Validation loss uses classification CE only.
