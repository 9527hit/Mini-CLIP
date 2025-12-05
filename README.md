# Mini-CLIP Tutorial

This is a Mini-CLIP implementation for educational purposes, demonstrating the core "alignment" mechanism in VLA (Vision-Language-Action) models.

## Principles
- **Image Encoder**: ResNet-18 (Pre-trained)
- **Text Encoder**: DistilBERT (Pre-trained)
- **Dataset**: CIFAR-10 (Auto-generated image-text pairs)
- **Loss**: Contrastive Loss (InfoNCE)

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Model
This will automatically download CIFAR-10 dataset and start training.
```bash
python train.py
```
*GPU training is recommended.*

### 3. Inference (Zero-Shot)
Run the inference script after training.
```bash
python inference.py
```
