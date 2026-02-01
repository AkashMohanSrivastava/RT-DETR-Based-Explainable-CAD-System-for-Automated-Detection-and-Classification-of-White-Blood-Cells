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

### 1. RT-DETR-L (Large)
- **Backbone**: ResNet-50
- **Training Mode**: Pretrained (fine-tuning)
- **Model File**: `rtdetr-l.pt`

### 2. RT-DETR-X (Extra Large)
- **Backbone**: ResNet-101
- **Training Mode**: Pretrained (fine-tuning)
- **Model File**: `rtdetr-x.pt`

### 3. RT-DETR-MobileNet
- **Backbone**: MobileNetV3-Small (lightweight)
- **Training Mode**: From scratch
- **Config File**: `rtdetr_mobilenetv3.yaml`

### 4. RT-DETR-ShuffleNet
- **Backbone**: ShuffleNetV2-Small (lightweight)
- **Training Mode**: From scratch
- **Config File**: `rtdetr_shufflenetv2.yaml`

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

| Model | Backbone | Pretrained | Accuracy | Inference (ms) | Training (s) |
|-------|----------|------------|----------|----------------|--------------|
| **RT-DETR-L** | ResNet-50 | Yes | **96.40%** | 44.37 | 3382.3 |
| RT-DETR-ShuffleNet | ShuffleNetV2-Small | No | 84.80% | 31.41 | 2330.4 |
| RT-DETR-X | ResNet-101 | Yes | 81.60% | 53.54 | 1969.4 |
| RT-DETR-MobileNet | MobileNetV3-Small | No | 62.80% | **29.72** | **1022.6** |

### Best Model: RT-DETR-L (ResNet-50)

The RT-DETR-L model with ResNet-50 backbone achieved the highest accuracy of **96.40%**.

#### Per-Class Performance (RT-DETR-L)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Basophil | 1.00 | 1.00 | 1.00 | 100 |
| Eosinophil | 1.00 | 0.97 | 0.98 | 100 |
| Lymphocyte | 0.90 | 0.96 | 0.93 | 100 |
| Monocyte | 1.00 | 0.89 | 0.94 | 100 |
| Neutrophil | 0.93 | 1.00 | 0.97 | 100 |
| **Weighted Avg** | **0.97** | **0.96** | **0.96** | **500** |

### Model Rankings

#### By Accuracy (Descending)
1. **RT-DETR-L**: 96.40%
2. RT-DETR-ShuffleNet: 84.80%
3. RT-DETR-X: 81.60%
4. RT-DETR-MobileNet: 62.80%

#### By Inference Speed (Ascending)
1. **RT-DETR-MobileNet**: 29.72ms
2. RT-DETR-ShuffleNet: 31.41ms
3. RT-DETR-L: 44.37ms
4. RT-DETR-X: 53.54ms

#### By Training Time (Ascending)
1. **RT-DETR-MobileNet**: 1022.6s (~17 min)
2. RT-DETR-X: 1969.4s (~33 min)
3. RT-DETR-ShuffleNet: 2330.4s (~39 min)
4. RT-DETR-L: 3382.3s (~56 min)

### Key Findings

- **Best Overall**: RT-DETR-L with ResNet-50 backbone achieves the highest accuracy (96.40%) with excellent per-class performance
- **Best for Real-Time**: RT-DETR-MobileNet offers the fastest inference (29.72ms) but with reduced accuracy
- **Best Balance**: RT-DETR-ShuffleNet provides a good trade-off between accuracy (84.80%) and speed (31.41ms)
- **Pretrained vs From-Scratch**: Pretrained models (RT-DETR-L) significantly outperform models trained from scratch on this dataset

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
