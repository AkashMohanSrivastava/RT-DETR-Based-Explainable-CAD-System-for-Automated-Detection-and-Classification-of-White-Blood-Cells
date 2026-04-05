# RT-DETR Based Explainable CAD System for Automated Detection and Classification of White Blood Cells

A Computer-Aided Diagnosis (CAD) system for automated detection and classification of White Blood Cells (WBCs) using RT-DETR (Real-Time Detection Transformer) with multiple backbone architectures.

## Overview

This project implements a deep learning-based system for automated White Blood Cell (WBC) classification using the RT-DETR object detection framework. The system is designed to assist medical professionals in analyzing blood smear images by automatically detecting and classifying different types of white blood cells.

### Key Features

- **Multiple Backbone Architectures**: Compare performance across different backbone networks (ResNet-50, ResNet-101, MobileNetV3, ShuffleNetV2)
- **RT-DETR Framework**: Utilizes Real-Time Detection Transformer for accurate object detection
- **Flexible Training**: Support for both full dataset and sampled dataset training
- **Pretrained & From-Scratch Training**: Options for fine-tuning pretrained models or training from scratch
- **Comprehensive Evaluation**: Detailed metrics including accuracy, balanced accuracy, MCC, Cohen's Kappa, mAP@0.50, mAP@0.50:0.95, confusion matrices, and per-class classification reports
- **Explainability**: GradCAM and ScoreCAM visualizations to highlight which image regions drive each model's predictions
- **Dataset Visualization**: Standalone notebook to inspect polygon segmentation annotations and bounding boxes per WBC class

## Dataset

### Raabin-WBC Dataset

This project uses the **Raabin-WBC** dataset, a large-scale dataset of white blood cell images for classification tasks.

#### Dataset Structure

```
Raabin_datsets_withlabels/
├── Train/
│   ├── images/
│   │   ├── Basophil/
│   │   ├── Eosinophil/
│   │   ├── Lymphocyte/
│   │   ├── Monocyte/
│   │   └── Neutrophil/
│   └── labels/
│       ├── Basophil/
│       ├── Eosinophil/
│       ├── Lymphocyte/
│       ├── Monocyte/
│       └── Neutrophil/
└── val/
    ├── images/
    │   ├── Basophil/
    │   ├── Eosinophil/
    │   ├── Lymphocyte/
    │   ├── Monocyte/
    │   └── Neutrophil/
    └── labels/
        ├── Basophil/
        ├── Eosinophil/
        ├── Lymphocyte/
        ├── Monocyte/
        └── Neutrophil/
```

#### WBC Classes

The dataset contains 5 types of white blood cells:

| Class ID | Cell Type | Description |
|----------|-----------|-------------|
| 0 | **Basophil** | Least common WBC type (~0.5-1%), involved in allergic reactions and inflammation |
| 1 | **Eosinophil** | Combat parasites and participate in allergic responses (~1-4%) |
| 2 | **Lymphocyte** | Key cells of the adaptive immune system (~20-40%), includes T-cells and B-cells |
| 3 | **Monocyte** | Largest WBC type (~2-8%), differentiate into macrophages and dendritic cells |
| 4 | **Neutrophil** | Most abundant WBC type (~55-70%), first responders to bacterial infections |

#### Dataset Statistics

- **Training Images**: ~10,000+ images across 5 classes
- **Validation Images**: 4,261 images (Basophil: 71, Eosinophil: 305, Lymphocyte: 1,017, Monocyte: 217, Neutrophil: 2,651)
- **Image Format**: JPG/JPEG
- **Label Format**: YOLO polygon segmentation format (class_id followed by normalized polygon vertex coordinates)

## Project Structure

```
RT-DETR-Based-Explainable-CAD-System/
├── README.md
├── Train_RT-DETR_L.ipynb                # Training with RT-DETR-L (ResNet-50)
├── Train_RT-DETR_X.ipynb                # Training with RT-DETR-X (ResNet-101)
├── Train_RT-DETR_MobileNet.ipynb        # Training with MobileNetV3 backbone
├── Train_RT-DETR_ShuffleNet.ipynb       # Training with ShuffleNetV2 backbone
├── Visualize_Dataset.ipynb              # Dataset visualization (standalone)
├── Compare_and_Visualize_Results.ipynb  # Results comparison and visualization
├── compare_utils.py                     # Shared utilities for the Compare notebook
├── training_utils.py                    # Shared utilities for training notebooks
├── rtdetr_mobilenetv3.yaml              # MobileNetV3 model configuration
├── rtdetr_shufflenetv2.yaml             # ShuffleNetV2 model configuration
├── rtdetr-l.pt                          # RT-DETR-L pretrained weights
├── rtdetr-x.pt                          # RT-DETR-X pretrained weights
├── yolo26n.pt                           # Lightweight YOLO model weights
├── runs/                                # Ultralytics default run output directory
└── output/
    ├── training_runs/                   # Training checkpoints and logs
    ├── checkpoints/                     # Model checkpoints
    ├── results/                         # Evaluation results (JSON and plots)
    └── data_full.yaml                   # Full dataset configuration
```

## Model Architectures

This project implements RT-DETR (Real-Time Detection Transformer) with four different backbone architectures. RT-DETR combines CNN backbones with transformer encoders for efficient object detection.

### RT-DETR Framework Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RT-DETR Architecture                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Input   │───>│   Backbone   │───>│   Hybrid     │───>│   RT-DETR    │  │
│  │  Image   │    │   (CNN)      │    │   Encoder    │    │   Decoder    │  │
│  └──────────┘    └──────────────┘    └──────────────┘    └──────────────┘  │
│                         │                   │                    │          │
│                         v                   v                    v          │
│                  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│                  │ Multi-scale │    │    AIFI     │    │   Detection    │  │
│                  │  Features   │    │ Transformer │    │     Heads      │  │
│                  │  P3,P4,P5   │    │   + FPN     │    │  (cls + bbox)  │  │
│                  └─────────────┘    └─────────────┘    └─────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 1. RT-DETR-L (Large) - ResNet-50 Backbone

- **Backbone**: ResNet-50
- **Training Mode**: Pretrained (fine-tuning from COCO weights)
- **Model File**: `rtdetr-l.pt`
- **Parameters**: ~32M

#### ResNet-50 Architecture

ResNet (Residual Network) introduced skip connections to enable training of very deep networks by addressing the vanishing gradient problem.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ResNet-50 Architecture                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input (640x640x3)                                                          │
│         │                                                                   │
│         v                                                                   │
│  ┌─────────────────┐                                                        │
│  │   Conv 7x7/2    │──> 320x320x64                                          │
│  │   MaxPool 3x3/2 │──> 160x160x64                                          │
│  └─────────────────┘                                                        │
│         │                                                                   │
│         v                                                                   │
│  ┌─────────────────┐     ┌───────────────────────────────┐                  │
│  │   Stage 1       │     │  Bottleneck Block (x3)        │                  │
│  │   (conv2_x)     │────>│  ┌─────┐  ┌─────┐  ┌─────┐   │                  │
│  │   160x160x256   │     │  │1x1  │─>│3x3  │─>│1x1  │─┐ │                  │
│  └─────────────────┘     │  │ 64  │  │ 64  │  │256  │ │ │                  │
│         │                │  └─────┘  └─────┘  └─────┘ │ │  Skip            │
│         v                │         +──────────────────┘ │  Connection      │
│  ┌─────────────────┐     └───────────────────────────────┘                  │
│  │   Stage 2       │                                                        │
│  │   (conv3_x)     │──> 80x80x512   (x4 blocks) ──> P3                      │
│  └─────────────────┘                                                        │
│         │                                                                   │
│         v                                                                   │
│  ┌─────────────────┐                                                        │
│  │   Stage 3       │                                                        │
│  │   (conv4_x)     │──> 40x40x1024  (x6 blocks) ──> P4                      │
│  └─────────────────┘                                                        │
│         │                                                                   │
│         v                                                                   │
│  ┌─────────────────┐                                                        │
│  │   Stage 4       │                                                        │
│  │   (conv5_x)     │──> 20x20x2048  (x3 blocks) ──> P5                      │
│  └─────────────────┘                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Key Features of ResNet-50:
- **Residual Learning**: Skip connections allow gradients to flow directly through the network
- **Bottleneck Design**: 1x1 → 3x3 → 1x1 convolutions reduce computational cost
- **Batch Normalization**: Applied after each convolution for stable training
- **50 Layers**: 1 conv + 3+4+6+3 bottleneck blocks (each with 3 conv layers)

---

### 2. RT-DETR-X (Extra Large) - ResNet-101 Backbone

- **Backbone**: ResNet-101
- **Training Mode**: Pretrained (fine-tuning from COCO weights)
- **Model File**: `rtdetr-x.pt`
- **Parameters**: ~67M

#### ResNet-101 Architecture

ResNet-101 extends ResNet-50 with deeper feature extraction, particularly in Stage 3.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ResNet-101 vs ResNet-50 Comparison                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Layer          │  ResNet-50   │  ResNet-101  │  Output Size                │
│  ───────────────┼──────────────┼──────────────┼─────────────                │
│  conv1          │  7x7, 64     │  7x7, 64     │  320x320                    │
│  ───────────────┼──────────────┼──────────────┼─────────────                │
│  conv2_x        │  3 blocks    │  3 blocks    │  160x160x256                │
│  ───────────────┼──────────────┼──────────────┼─────────────                │
│  conv3_x (P3)   │  4 blocks    │  4 blocks    │  80x80x512                  │
│  ───────────────┼──────────────┼──────────────┼─────────────                │
│  conv4_x (P4)   │  6 blocks    │  23 blocks   │  40x40x1024   <── Deeper!   │
│  ───────────────┼──────────────┼──────────────┼─────────────                │
│  conv5_x (P5)   │  3 blocks    │  3 blocks    │  20x20x2048                 │
│  ───────────────┴──────────────┴──────────────┴─────────────                │
│                                                                             │
│  Total Layers:    50 layers      101 layers                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Key Differences from ResNet-50:
- **Deeper Stage 3**: 23 bottleneck blocks vs 6 (more feature extraction capacity)
- **Better Feature Representation**: Captures more complex patterns
- **Higher Computational Cost**: ~2x more parameters than ResNet-50
- **Recommended for**: High-accuracy requirements where speed is secondary

---

### 3. RT-DETR-MobileNet - MobileNetV3-Small Backbone

- **Backbone**: MobileNetV3-Small (custom implementation)
- **Training Mode**: From scratch
- **Config File**: `rtdetr_mobilenetv3.yaml`
- **Parameters**: ~3M (backbone only)

#### MobileNetV3-Small Architecture

MobileNetV3 is designed for mobile and edge devices, using depthwise separable convolutions and squeeze-and-excitation blocks for efficiency.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MobileNetV3-Small Architecture                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input (512x512x3)                                                          │
│         │                                                                   │
│         v                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Stem: Conv 3x3/2 ──> 256x256x16                                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│         │                                                                   │
│         v                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Stage 1-2: DWConv blocks ──> 128x128x24                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│         │                                                                   │
│         v                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Stage 3 (P3): Expand(72) → DWConv 5x5/2 → Project(40) → C2f        │    │
│  │  Output: 64x64x40                                                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│         │                                                                   │
│         v                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Stage 4 (P4): Expand(120) → DWConv 5x5/2 → Project(80) → C2f(x2)   │    │
│  │  Output: 32x32x80                                                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│         │                                                                   │
│         v                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Stage 5 (P5): Expand(240) → DWConv 5x5/2 → Project(160) → C2f      │    │
│  │  Output: 16x16x160                                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    Depthwise Separable Convolution                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Standard Conv (3x3, C_in → C_out):                                         │
│  Params = 3 × 3 × C_in × C_out                                              │
│                                                                             │
│  Depthwise Separable Conv:                                                  │
│  ┌──────────────┐    ┌──────────────┐                                       │
│  │  Depthwise   │───>│  Pointwise   │                                       │
│  │  3x3 × C_in  │    │  1x1 × C_out │                                       │
│  └──────────────┘    └──────────────┘                                       │
│  Params = 3 × 3 × C_in + C_in × C_out                                       │
│                                                                             │
│  Reduction: ~8-9x fewer parameters!                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Key Features of MobileNetV3:
- **Depthwise Separable Convolutions**: 8-9x parameter reduction vs standard conv
- **Inverted Residuals**: Expand → Depthwise → Project (thin-thick-thin)
- **5x5 Depthwise Kernels**: Larger receptive field with minimal cost increase
- **Hard-Swish Activation**: Efficient approximation of Swish activation
- **Optimized for Mobile**: Designed for <100ms inference on mobile CPUs

#### Our Custom Configuration (`rtdetr_mobilenetv3.yaml`):

| Stage | Expansion | Kernel | Output Channels | C2f Blocks |
|-------|-----------|--------|-----------------|------------|
| P3/8  | 72        | 5x5    | 40              | 1          |
| P4/16 | 120       | 5x5    | 80              | 2          |
| P5/32 | 240       | 5x5    | 160             | 1          |

---

### 4. RT-DETR-ShuffleNet - ShuffleNetV2-Small Backbone

- **Backbone**: ShuffleNetV2-Small (custom implementation)
- **Training Mode**: From scratch
- **Config File**: `rtdetr_shufflenetv2.yaml`
- **Parameters**: ~2.5M (backbone only)

#### ShuffleNetV2-Small Architecture

ShuffleNetV2 achieves efficiency through channel shuffle operations and balanced channel splits, optimized for actual hardware inference speed.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ShuffleNetV2-Small Architecture                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input (512x512x3)                                                          │
│         │                                                                   │
│         v                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Stem: Conv 3x3/2 ──> 256x256x24                                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│         │                                                                   │
│         v                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Stage 2 (P2): DWConv/2 → 1x1 → C2f(x2) ──> 128x128x48              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│         │                                                                   │
│         v                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Stage 3 (P3): DWConv/2 → 1x1 → C2f(x2) ──> 64x64x96                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│         │                                                                   │
│         v                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Stage 4 (P4): DWConv/2 → 1x1 → C2f(x3) ──> 32x32x192               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│         │                                                                   │
│         v                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Stage 5 (P5): DWConv/2 → 1x1 → C2f(x2) ──> 16x16x384               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                       Channel Shuffle Operation                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input Channels: [A1 A2 A3 B1 B2 B3]  (2 groups × 3 channels)               │
│                                                                             │
│  Step 1 - Reshape:    ┌─────────────┐                                       │
│                       │ A1  A2  A3  │                                       │
│                       │ B1  B2  B3  │                                       │
│                       └─────────────┘                                       │
│                                                                             │
│  Step 2 - Transpose:  ┌─────────────┐                                       │
│                       │ A1  B1      │                                       │
│                       │ A2  B2      │                                       │
│                       │ A3  B3      │                                       │
│                       └─────────────┘                                       │
│                                                                             │
│  Step 3 - Flatten:    [A1 B1 A2 B2 A3 B3]  (shuffled!)                      │
│                                                                             │
│  Purpose: Enable cross-group information flow without extra computation     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Key Features of ShuffleNetV2:
- **Channel Shuffle**: Zero-cost operation for cross-channel information mixing
- **Channel Split**: Half channels for identity, half for transformation
- **Balanced Channels**: Equal channel widths at each layer for memory efficiency
- **No Group Convolutions**: Removed to improve memory access patterns
- **Practical Speed**: Designed based on actual inference benchmarks, not just FLOPs

#### Our Custom Configuration (`rtdetr_shufflenetv2.yaml`):

| Stage | Input Ch | Output Ch | C2f Blocks | Stride |
|-------|----------|-----------|------------|--------|
| P2/4  | 24       | 48        | 2          | 2      |
| P3/8  | 48       | 96        | 2          | 2      |
| P4/16 | 96       | 192       | 3          | 2      |
| P5/32 | 192      | 384       | 2          | 2      |

---

### RT-DETR Head Architecture (Common to All Backbones)

All backbone outputs feed into the same RT-DETR head architecture:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RT-DETR Hybrid Encoder                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Backbone Features                                                          │
│  ┌─────┐ ┌─────┐ ┌─────┐                                                    │
│  │ P3  │ │ P4  │ │ P5  │                                                    │
│  └──┬──┘ └──┬──┘ └──┬──┘                                                    │
│     │       │       │                                                       │
│     │       │       v                                                       │
│     │       │  ┌─────────────────┐                                          │
│     │       │  │  AIFI (Intra-   │  Attention-based Intra-scale             │
│     │       │  │  scale Fusion)  │  Feature Interaction                     │
│     │       │  │  512 dims,      │  (Transformer encoder on P5)             │
│     │       │  │  4 heads        │                                          │
│     │       │  └────────┬────────┘                                          │
│     │       │           │                                                   │
│     │       │           v                                                   │
│     │       │  ┌─────────────────┐                                          │
│     │       └─>│    Top-Down     │  Feature Pyramid Network                 │
│     │          │    Path (FPN)   │  Upsample + Concat + RepC3               │
│     └─────────>│                 │                                          │
│                └────────┬────────┘                                          │
│                         │                                                   │
│                         v                                                   │
│                ┌─────────────────┐                                          │
│                │   Bottom-Up     │  Path Aggregation Network                │
│                │   Path (PAN)    │  Downsample + Concat + RepC3             │
│                └────────┬────────┘                                          │
│                         │                                                   │
│                         v                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     RTDETRDecoder                                   │    │
│  │  - 300 object queries                                               │    │
│  │  - 4 decoder layers                                                 │    │
│  │  - 8 attention heads                                                │    │
│  │  - 3 feature levels (P3, P4, P5)                                    │    │
│  │  - Classification head (nc classes)                                 │    │
│  │  - Bounding box regression head                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Architecture Comparison Summary

| Feature | ResNet-50 | ResNet-101 | MobileNetV3 | ShuffleNetV2 |
|---------|-----------|------------|-------------|--------------|
| **Type** | Deep Residual | Deep Residual | Inverted Residual | Channel Shuffle |
| **Params (backbone)** | ~23M | ~42M | ~3M | ~2.5M |
| **Key Innovation** | Skip connections | Deeper Stage 3 | Depthwise Sep Conv | Channel Shuffle |
| **Pretrained** | Yes (COCO) | Yes (COCO) | No | No |
| **Target Use** | High accuracy | Maximum accuracy | Mobile/Edge | Mobile/Edge |
| **Inference Speed** | Medium | Slow | Fast | Fast |

### External Architecture References

For detailed visual diagrams, refer to the original papers:

- **ResNet**: [Deep Residual Learning (He et al., 2015)](https://arxiv.org/abs/1512.03385)
- **MobileNetV3**: [Searching for MobileNetV3 (Howard et al., 2019)](https://arxiv.org/abs/1905.02244)
- **ShuffleNetV2**: [ShuffleNet V2: Practical Guidelines (Ma et al., 2018)](https://arxiv.org/abs/1807.11164)
- **RT-DETR**: [DETRs Beat YOLOs on Real-time Object Detection (Lv et al., 2023)](https://arxiv.org/abs/2304.08069)

## Requirements

```
torch>=2.0.0
torchvision
ultralytics>=8.0.0
numpy
scikit-learn
tqdm
pyyaml
matplotlib
seaborn
pillow
timm
opencv-python
pandas
```

## Installation

```bash
pip install -U ultralytics torch torchvision pillow tqdm scikit-learn seaborn timm pyyaml opencv-python pandas
```

## Usage

### Configuration Options

Each training notebook includes configurable options in the **Configuration** cell:

```python
# Dataset Mode
USE_FULL_DATASET = True   # True: use all images, False: use sampled subset

# Sample sizes (only used when USE_FULL_DATASET=False)
TRAIN_SAMPLE_SIZE = 100   # Training samples per class
VAL_SAMPLE_SIZE = 20      # Validation samples per class

# Training Hyperparameters
TRAINING_CONFIG = {
    "epochs": 20,
    "imgsz": 640,
    "batch": 8,
    "lr0": 0.0001,
    ...
}
```

### Training a Model

1. Open the desired training notebook (e.g., `Train_RT-DETR_L.ipynb`)
2. Configure the dataset path in the **Configuration** cell:
   ```python
   DATA_ROOT = r"path/to/Raabin_datsets_withlabels"
   ```
3. Set `USE_FULL_DATASET = True` for full dataset or `False` for sampled training
4. Run all cells to train and evaluate the model

### Visualizing the Dataset

Run `Visualize_Dataset.ipynb` standalone (no training required) to:
- View polygon segmentation annotations overlaid on images for each WBC class
- Inspect random samples with bounding boxes derived from polygon labels

### Comparing Results

Use `Compare_and_Visualize_Results.ipynb` to:
- Load results from all trained models
- Compare accuracy, balanced accuracy, MCC, Cohen's Kappa, mAP@0.50, and mAP@0.50:0.95
- Visualize confusion matrices and per-class F1 scores
- Plot training and validation loss curves
- Inspect detection samples per class for each model
- Generate GradCAM and ScoreCAM explainability visualizations
- Compute FLOPs and parameter counts per model

## Output Files

After training, the following files are generated:

```
output/
├── training_runs/
│   └── MODEL_NAME_TIMESTAMP/
│       ├── weights/
│       │   ├── best.pt          # Best model weights
│       │   └── last.pt          # Last epoch weights
│       ├── results.csv          # Training metrics
│       └── args.yaml            # Training arguments
└── results/
    └── MODEL_NAME_results.json  # Evaluation results
```

### Results JSON Structure

```json
{
    "model_name": "RT-DETR-L",
    "backbone": "ResNet-50",
    "accuracy": 0.85,
    "avg_inference_time_ms": 35.5,
    "confusion_matrix": [...],
    "classification_report": {...},
    "training_time_s": 1234.5
}
```

## Training Modes

### Full Dataset Training
- Uses all available images from both `Train/` and `val/` directories
- Recommended for final model training
- Set `USE_FULL_DATASET = True`

### Sampled Dataset Training
- Uses a subset of images for faster experimentation
- Samples from both `Train/` and `val/` directories independently
- Configure `TRAIN_SAMPLE_SIZE` and `VAL_SAMPLE_SIZE`
- Set `USE_FULL_DATASET = False`

## Experimental Results

### Model Comparison

All models were trained on the Raabin-WBC dataset and evaluated on the full validation set (4,261 images; class distribution: Basophil 71, Eosinophil 305, Lymphocyte 1,017, Monocyte 217, Neutrophil 2,651). Epochs shown are actual epochs trained (early stopping active for all models; configured max: 20 for pretrained, 50 for from-scratch).

| Model | Backbone | Pretrained | Max Epochs | Actual Epochs | Accuracy | Bal. Acc | MCC | mAP@0.50 | mAP@0.50:0.95 | Inference (ms) | Training (s) |
|-------|----------|------------|-----------|---------------|----------|----------|-----|----------|----------------|----------------|--------------|
| **RT-DETR-L** | ResNet-50 | Yes | 20 | 19 | **99.04%** | 0.987 | 0.983 | **0.957** | **0.898** | 45.05 | 11231.6 |
| RT-DETR-X | ResNet-101 | Yes | 20 | 10 | 97.23% | 0.966 | 0.950 | 0.847 | 0.787 | 53.47 | 14768.5 |
| RT-DETR-ShuffleNet | ShuffleNetV2-Small | No | 50 | 15 | 94.51% | 0.936 | 0.904 | 0.823 | 0.770 | **25.60** | 4360.1 |
| RT-DETR-MobileNet | MobileNetV3-Small | No | 50 | 23 | 93.73% | 0.943 | 0.893 | 0.811 | 0.757 | 26.29 | 6642.5 |

### Best Model: RT-DETR-L (ResNet-50)

The RT-DETR-L model with ResNet-50 backbone achieved the highest accuracy of **99.04%** (evaluated on 4,261 validation images).

#### Per-Class Performance (RT-DETR-L)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Basophil | 1.00 | 1.00 | 1.00 | 71 |
| Eosinophil | 0.94 | 0.99 | 0.97 | 305 |
| Lymphocyte | 0.99 | 1.00 | 0.99 | 1017 |
| Monocyte | 0.98 | 0.95 | 0.97 | 217 |
| Neutrophil | 1.00 | 0.99 | 0.99 | 2651 |
| **Weighted Avg** | **0.99** | **0.99** | **0.99** | **4261** |

### Model Rankings

#### By Accuracy (Descending)
1. **RT-DETR-L**: 99.04% (19 epochs)
2. RT-DETR-X: 97.23% (10 epochs)
3. RT-DETR-ShuffleNet: 94.51% (15 epochs)
4. RT-DETR-MobileNet: 93.73% (23 epochs)

#### By Inference Speed (Ascending)
1. **RT-DETR-ShuffleNet**: 25.60ms
2. RT-DETR-MobileNet: 26.29ms
3. RT-DETR-L: 45.05ms
4. RT-DETR-X: 53.47ms

#### By Training Time (Ascending)
1. **RT-DETR-ShuffleNet**: 4360.1s (~73 min, 15 epochs)
2. RT-DETR-MobileNet: 6642.5s (~111 min, 23 epochs)
3. RT-DETR-L: 11231.6s (~187 min, 19 epochs)
4. RT-DETR-X: 14768.5s (~246 min, 10 epochs)

#### By mAP@0.50 (Descending)
1. **RT-DETR-L**: 0.957
2. RT-DETR-X: 0.847
3. RT-DETR-ShuffleNet: 0.823
4. RT-DETR-MobileNet: 0.811

### Key Findings

- **Best Overall**: RT-DETR-L with ResNet-50 backbone achieves the highest accuracy (99.04%), MCC of 0.983, and mAP@0.50 of 0.957, running 19 of a configured 20 epochs before early stopping
- **Pretrained Advantage**: Both pretrained models (RT-DETR-L, RT-DETR-X) outperform the from-scratch models on accuracy and mAP; RT-DETR-L converges in 19 epochs vs 23 for MobileNet despite starting from better initial weights
- **Early Stopping**: All four models stopped before their configured epoch budget — RT-DETR-X stopped at only 10/20 epochs, suggesting good generalization for a ResNet-101 model
- **Best for Edge Deployment**: RT-DETR-ShuffleNet offers the fastest inference (25.60ms) with competitive accuracy (94.51%), making it suitable for resource-constrained environments
- **Lightweight Efficiency**: Both MobileNet and ShuffleNet backbones reach >93% accuracy with ~12–15M total parameters and <33 GFLOPs, compared to RT-DETR-L's 32.82M parameters and 70.36 GFLOPs
- **Monocyte is the Hardest Class**: Across all models, Monocyte shows the highest misclassification rate — its kidney-shaped nucleus is sometimes confused with Lymphocyte
- **Explainability**: GradCAM and ScoreCAM maps confirm that all models focus on the cell nucleus region, validating that decisions are biologically meaningful

### Visualization Outputs

`Compare_and_Visualize_Results.ipynb` generates the following in `output/results/` and `output/`:

| File | Description |
|------|-------------|
| `model_performance.png` | Bar charts comparing accuracy, balanced accuracy, MCC, Cohen's Kappa, mAP@0.50, and mAP@0.50:0.95 |
| `model_efficiency.png` / `model_efficiency2.png` | Inference time, training time, and model size comparison |
| `confusion_matrices.png` | Confusion matrices for all models |
| `f1_comparison.png` | Per-class F1 score comparison across models |
| `loss_curves.png` | Training and validation loss curves |
| `detection_samples_all_models.png` | Sample detections with bounding boxes for each model and class |
| `gradcam_RT-DETR-*.png` | GradCAM activation maps per model |
| `scorecam_RT-DETR-*.png` | ScoreCAM activation maps per model |
| `gradcam_vs_scorecam_RT-DETR-*.png` | Side-by-side GradCAM vs ScoreCAM comparison per model |
| `hyperparameter_comparison.png` | Hyperparameter configuration table across all models |

`Visualize_Dataset.ipynb` visualizes the training data inline (no files saved):
- Polygon segmentation annotations per class (green outline + red dashed bounding box)
- Random samples with bounding boxes across all classes


## FLOPs and Parameter Count

Computational cost for a single forward pass at input resolution 512×512:

| Model | Backbone | Total Params (M) | GFLOPs | Accuracy |
|-------|----------|-----------------|--------|----------|
| RT-DETR-MobileNet | MobileNetV3-Small | 12.44 | 28.50 | 93.73% |
| RT-DETR-ShuffleNet | ShuffleNetV2-Small | 14.87 | 32.23 | 94.51% |
| RT-DETR-L | ResNet-50 | 32.82 | 70.36 | 99.04% |
| RT-DETR-X | ResNet-101 | ~67 | ~150 | 97.23% |

The two lightweight models offer a 2.3–2.5x reduction in FLOPs and a 2.2–2.6x reduction in parameters compared to RT-DETR-L.

## Explainability

To ensure transparency and clinical trustworthiness, `Compare_and_Visualize_Results.ipynb` generates two types of Class Activation Maps (CAM) for all four models:

- **GradCAM** (Gradient-weighted Class Activation Mapping): uses gradients flowing back into the last backbone feature map to produce a saliency heatmap. Target layers are model-specific (Conv for ResNet variants, C2f for lightweight models).
- **ScoreCAM** (Score-weighted Class Activation Mapping): a gradient-free alternative that weights activation maps by their contribution to the prediction score, reducing noise from gradient instability.

Both methods confirm that the models focus on the **cell nucleus and cytoplasm regions** rather than background, which is biologically consistent with how pathologists identify WBC types.

## License

This project is for educational and research purposes.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLO and RT-DETR implementations
- Raabin-WBC dataset creators for providing the white blood cell image dataset
