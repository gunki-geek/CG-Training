# CGT Early-Exit ResNet on CIFAR-100 (HardCGT / SoftCGT)

This repository contains training and evaluation code for **Confidence-Gated Training (CGT)** on an **early-exit ResNet-18** with **4 exits** for **CIFAR-100**.

- **SoftCGT training**: per-sample residual gating using GT-class probability.
- **HardCGT training**: per-sample binary gate (correct + confident) that blocks deeper gradients when earlier exits succeed.
- **Evaluation**: per-exit accuracy, early-exit accuracy + exit distribution for chosen inference thresholds, and average inference time per sample.

## Files
- `resnet.py`: early-exit ResNet-18 definition. `forward()` returns logits as `[out4, out3, out2, out1]` (deep→shallow) and features. (We reorder to shallow→deep in eval/training helpers.)  
- `train_soft_cgt.py`: SoftCGT training on CIFAR-100 using Resnet18 as backbone.
- `train_hard_cgt.py`: HardCGT training on CIFAR-100 using Resnet18 as backbone.
- `eval.py`: evaluation on CIFAR-100 (per-exit accuracy, early-exit behavior, and inference time).

## Requirements
- Python 3.9+
- PyTorch + torchvision
- NVIDIA GPU recommended for speed (CPU also works)

Install (example):
```bash
pip install torch torchvision

```
## SoftCGT Training

SoftCGT uses a **soft confidence-gated mechanism** where the training signal from deeper exits is modulated using the **ground-truth class probability** of earlier exits. This allows gradients to flow smoothly while prioritizing exits that fail to confidently classify a sample.


### SoftCGT Training Command

```bash
python3 train_soft_cgt.py \
  --data-dir ./data \
  --batch-size 500 \
  --epochs 300 \
  --lr 0.1 \
  --weight-decay 5e-4 \
  --tau-train 0.9 \
  --tau-eval 0.9 \
  --device cuda
```

## HardCGT Training
HardCGT employs a binary confidence-gated training strategy. If an early exit predicts the correct class with confidence above the threshold, gradients from deeper exits are blocked for that sample. This enforces strict exit specialization but makes HardCGT more sensitive to the choice of the confidence threshold.

### HardCGT Training Command

```bash
python3 train_hard_cgt.py \
  --data-dir ./data \
  --batch-size 500 \
  --epochs 300 \
  --lr 0.1 \
  --weight-decay 5e-4 \
  --tau-train 0.9 \
  --tau-eval 0.9 \
  --device cuda
```

## Evaluate best models
To reproduce the best results of our experiments run the following commands

### Best HardCGT model
```bash
python3 eval_hardCGT.py
```
### Best SoftCGT model
```bash
python3 eval_softCGT.py
```

## Evaluation
The evaluation script measures accuracy, early-exit behavior, and efficiency of a trained model.

Specifically, it reports:

Per-exit accuracy: each exit evaluated independently on all test samples.
Early-exit accuracy and exit distribution under confidence-based inference.
Average inference time per sample, properly synchronized on GPU.

### Evaluation Command (Single Threshold)
replace the argument to ckpt with the actual save path of your trained model
```bash
python3 eval.py \
  --ckpt checkpoints/best_resnet18_att_softcgt_cifar100.pth \
  --tau 0.9 \
  --device cuda
```

### Evaluation Command (Per-Exit Thresholds)
replace the argument to ckpt with the actual save path of your trained model
```bash
python3 eval.py \
  --ckpt checkpoints/best_resnet18_att_softcgt_cifar100.pth \
  --taus 0.95,0.90,0.85 \
  --batch-size 512 \
  --device cuda
```
