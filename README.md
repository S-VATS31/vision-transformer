# Vision Transformer for ImageNet-1K

This repository contains a PyTorch implementation of a Vision Transformer (ViT) model for image classification on the ImageNet-1K dataset. The model integrates advanced techniques such as Grouped Query Attention (GQA), Rotary Positional Embeddings (RoPE), and robust data augmentation strategies to achieve high performance.

## Table of Contents

- Introduction
- Model Architecture
  - Patch Embeddings
  - RMSNorm
  - Rotary Positional Embeddings (RoPE)
  - Grouped Query Attention (GQA)
  - Feed-Forward Network (FFN)
  - Transformer Encoder
- Training Details
  - Data Augmentation
  - Optimization
  - Learning Rate Scheduling
- Usage
  - Installation
  - Training
  - Resuming Training
- Results
- Conclusion

## Introduction

Vision Transformers (ViTs) have demonstrated state-of-the-art results on a wide range of vision tasks. This implementation is tailored for ImageNet-1K classification with 1,000 classes and includes innovations to boost both accuracy and efficiency.

## Model Architecture

### Patch Embeddings

The `PatchEmbeddings` layer divides the image into fixed-size patches and projects them into a lower-dimensional embedding space.

- Divides input into non-overlapping square patches
- Flattens and linearly projects each patch
- Adds a learnable CLS token for classification

### RMSNorm

Root Mean Square Layer Normalization offers a lightweight alternative to LayerNorm:

- Normalizes inputs using root mean square
- Has a single learnable scaling parameter

### Rotary Positional Embeddings (RoPE)

RoPE enhances attention with 2D positional awareness:

- Applies rotary transformations in both x and y dimensions
- Uses sine and cosine rotations to embed spatial positions
- Efficient and compatible with FlashAttention

### Grouped Query Attention (GQA)

Grouped Query Attention increases efficiency without sacrificing expressiveness:

- Divides attention heads into query groups
- Shares key-value pairs among queries in the same group
- Reduces memory usage and compute

### Feed-Forward Network (FFN)

The FFN enhances feature representations through non-linear transformation:

- Two linear layers with a GELU activation in between
- Adds non-linearity and depth to model representations

### Transformer Encoder

Each encoder block consists of:

- RMSNorm → GQA → Dropout → Residual
- RMSNorm → FFN → Dropout → Residual

This structure promotes gradient flow and regularization.

## Training Details

### Data Augmentation

To improve generalization and reduce overfitting, the following techniques are used:

- Mixup: Blends images and labels
- CutMix: Mixes patches between images
- Random Erasing: Randomly removes parts of the image
- Color Jitter: Alters brightness, contrast, saturation, hue
- AutoAugment: Applies learned augmentation policies

### Optimization

- Optimizer: AdamW with weight decay  
- Gradient Clipping to prevent exploding gradients  
- Learning rate schedule: Linear Warmup + Cosine Annealing

### Learning Rate Scheduling

- Linear Warmup gradually increases the learning rate during early training steps  
- Cosine Annealing reduces the learning rate smoothly toward zero over time  

## Usage

### Installation

Install required dependencies:

```bash
pip install torch torchvision matplotlib tqdm tensorboard
