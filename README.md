# RT-DETR Based Explainable CAD System for Automated Detection and Classification of White Blood Cells

A Computer-Aided Diagnosis (CAD) system for automated detection and classification of White Blood Cells (WBCs) using RT-DETR (Real-Time Detection Transformer) with multiple backbone architectures.

## Overview

This project implements a deep learning-based system for automated White Blood Cell (WBC) classification using the RT-DETR object detection framework. The system is designed to assist medical professionals in analyzing blood smear images by automatically detecting and classifying different types of white blood cells.

### Key Features

- **Multiple Backbone Architectures**: Compare performance across different backbone networks (ResNet-50, ResNet-101, MobileNetV3, ShuffleNetV2)
- **RT-DETR Framework**: Utilizes Real-Time Detection Transformer for accurate object detection
- **Flexible Training**: Support for both full dataset and sampled dataset training
- **Pretrained & From-Scratch Training**: Options for fine-tuning pretrained models or training from scratch
- **Comprehensive Evaluation**: Detailed metrics including accuracy, confusion matrix, and per-class classification reports

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
- **Validation Images**: ~4,000+ images across 5 classes
- **Image Format**: JPG/JPEG
- **Label Format**: YOLO format (class_id, x_center, y_center, width, height)

## Project Structure

```
RT-DETR-Based-Explainable-CAD-System/
├── README.md
├── Train_RT-DETR_L.ipynb           # Training with RT-DETR-L (ResNet-50)
├── Train_RT-DETR_X.ipynb           # Training with RT-DETR-X (ResNet-101)
├── Train_RT-DETR_MobileNet.ipynb   # Training with MobileNetV3 backbone
├── Train_RT-DETR_ShuffleNet.ipynb  # Training with ShuffleNetV2 backbone
├── Compare_and_Visualize_Results.ipynb  # Results comparison and visualization
├── rtdetr_mobilenetv3.yaml         # MobileNetV3 model configuration
├── rtdetr_shufflenetv2.yaml        # ShuffleNetV2 model configuration
└── output/
    ├── training_runs/              # Training checkpoints and logs
    ├── results/                    # Evaluation results (JSON)
    ├── data_subset/                # Sampled dataset configuration
    └── data_full.yaml              # Full dataset configuration
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
```

## Installation

```bash
pip install -U ultralytics torch torchvision pillow tqdm scikit-learn seaborn timm pyyaml
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

### Comparing Results

Use `Compare_and_Visualize_Results.ipynb` to:
- Load results from all trained models
- Compare accuracy and inference times
- Visualize confusion matrices
- Generate comparative charts

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

All models were trained on the Raabin-WBC dataset and evaluated on 500 test samples (100 per class).

| Model | Backbone | Pretrained | Epochs | Accuracy | Inference (ms) | Training (s) |
|-------|----------|------------|--------|----------|----------------|--------------|
| **RT-DETR-L** | ResNet-50 | Yes | 5 | **96.80%** | 40.57 | 1300.4 |
| RT-DETR-MobileNet | MobileNetV3-Small | No | 15 | 94.40% | **26.46** | 1535.6 |
| RT-DETR-ShuffleNet | ShuffleNetV2-Small | No | 15 | 93.20% | 26.54 | 1517.7 |
| RT-DETR-X | ResNet-101 | Yes | 5 | 91.40% | 48.73 | 3152.4 |

### Best Model: RT-DETR-L (ResNet-50)

The RT-DETR-L model with ResNet-50 backbone achieved the highest accuracy of **96.80%**.

#### Per-Class Performance (RT-DETR-L)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Basophil | 0.98 | 1.00 | 0.99 | 100 |
| Eosinophil | 0.98 | 0.98 | 0.98 | 100 |
| Lymphocyte | 0.97 | 0.93 | 0.95 | 100 |
| Monocyte | 0.98 | 0.94 | 0.96 | 100 |
| Neutrophil | 0.93 | 0.99 | 0.96 | 100 |
| **Weighted Avg** | **0.97** | **0.97** | **0.97** | **500** |

### Model Rankings

#### By Accuracy (Descending)
1. **RT-DETR-L**: 96.80% (5 epochs)
2. RT-DETR-MobileNet: 94.40% (15 epochs)
3. RT-DETR-ShuffleNet: 93.20% (15 epochs)
4. RT-DETR-X: 91.40% (5 epochs)

#### By Inference Speed (Ascending)
1. **RT-DETR-MobileNet**: 26.46ms
2. RT-DETR-ShuffleNet: 26.54ms
3. RT-DETR-L: 40.57ms
4. RT-DETR-X: 48.73ms

#### By Training Time (Ascending)
1. **RT-DETR-L**: 1300.4s (~22 min, 5 epochs)
2. RT-DETR-ShuffleNet: 1517.7s (~25 min, 15 epochs)
3. RT-DETR-MobileNet: 1535.6s (~26 min, 15 epochs)
4. RT-DETR-X: 3152.4s (~53 min, 5 epochs)

### Key Findings

- **Best Overall**: RT-DETR-L with ResNet-50 backbone achieves the highest accuracy (96.80%) with only 5 epochs of training
- **Best for Real-Time**: RT-DETR-MobileNet offers the fastest inference (26.46ms) with strong accuracy (94.40%) after 15 epochs
- **Efficient Lightweight Models**: Both MobileNet and ShuffleNet backbones achieve >93% accuracy when trained for more epochs, making them suitable for edge deployment
- **Pretrained Advantage**: Pretrained models (RT-DETR-L) converge faster, achieving top accuracy in only 5 epochs
- **Training Trade-off**: Lightweight models trained from scratch require more epochs (15) but achieve competitive accuracy with faster inference

### Visualization Outputs

The comparison notebook generates the following visualizations in `output/results/`:
- `model_comparison.png` - Bar charts comparing accuracy, inference time, and training time
- `confusion_matrices.png` - Confusion matrices for all models
- `f1_comparison.png` - Per-class F1 score comparison across models
- `comparison_results.json` - Combined results in JSON format


## License

This project is for educational and research purposes.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLO and RT-DETR implementations
- Raabin-WBC dataset creators for providing the white blood cell image dataset
