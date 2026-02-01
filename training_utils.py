"""
Common training utilities for RT-DETR WBC Classification.

This module contains shared functions used across all training notebooks:
- Dataset creation and sampling
- Model training with checkpoint/resume support
- Model evaluation
- Results processing
"""

import os
import json
import random
import shutil
import time
from datetime import datetime

import numpy as np
import yaml
from tqdm import tqdm

from ultralytics import RTDETR
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def create_sampled_dataset(data_root, base_dir, classes, train_samples_per_class, val_samples_per_class, random_seed=42):
    """
    Create a sampled dataset from separate Train and val directories.

    Args:
        data_root: Root directory containing Train/ and val/ folders
        base_dir: Output directory for subset configuration
        classes: Dict mapping class names to class IDs
        train_samples_per_class: Number of training samples per class
        val_samples_per_class: Number of validation samples per class
        random_seed: Random seed for reproducibility

    Returns:
        Path to the generated data.yaml configuration file
    """
    if random_seed is not None:
        random.seed(random_seed)

    # Define paths
    subset_dir = os.path.join(base_dir, "data_subset")

    # Source paths - separate Train and val directories
    src_train_images = os.path.join(data_root, "Train", "images")
    src_train_labels = os.path.join(data_root, "Train", "labels")
    src_val_images = os.path.join(data_root, "val", "images")
    src_val_labels = os.path.join(data_root, "val", "labels")

    # Clean up existing subset directory
    if os.path.exists(subset_dir):
        shutil.rmtree(subset_dir)
    os.makedirs(subset_dir, exist_ok=True)

    # Lists to store image paths
    train_image_paths = []
    val_image_paths = []
    labels_corrected = 0

    total_train = 0
    total_val = 0

    print("Sampling from TRAINING dataset:")
    for cls_name, cls_id in classes.items():
        # --- Training data (from Train directory) ---
        src_cls_images = os.path.join(src_train_images, cls_name)
        src_cls_labels = os.path.join(src_train_labels, cls_name)

        if os.path.exists(src_cls_images):
            image_files = [f for f in os.listdir(src_cls_images) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            # Sample training images
            if len(image_files) > train_samples_per_class:
                image_files = random.sample(image_files, train_samples_per_class)

            for img_file in image_files:
                base_name = os.path.splitext(img_file)[0]
                original_img_path = os.path.join(src_cls_images, img_file)
                train_image_paths.append(original_img_path)

                # Correct label file if needed
                label_file = base_name + ".txt"
                label_path = os.path.join(src_cls_labels, label_file)

                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        lines = f.readlines()

                    new_lines = []
                    needs_correction = False
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) > 1:
                            if parts[0] != str(cls_id):
                                needs_correction = True
                                parts[0] = str(cls_id)
                            new_lines.append(' '.join(parts) + '\n')

                    if needs_correction:
                        with open(label_path, 'w') as f:
                            f.writelines(new_lines)
                        labels_corrected += 1

            total_train += len(image_files)
            print(f"  {cls_name}: {len(image_files)} images")

    print(f"\nSampling from VALIDATION dataset:")
    for cls_name, cls_id in classes.items():
        # --- Validation data (from val directory - NOT from Train) ---
        src_cls_val_images = os.path.join(src_val_images, cls_name)
        src_cls_val_labels = os.path.join(src_val_labels, cls_name)

        if os.path.exists(src_cls_val_images):
            val_files = [f for f in os.listdir(src_cls_val_images) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            # Sample validation images
            if len(val_files) > val_samples_per_class:
                val_files = random.sample(val_files, val_samples_per_class)

            for img_file in val_files:
                base_name = os.path.splitext(img_file)[0]
                original_img_path = os.path.join(src_cls_val_images, img_file)
                val_image_paths.append(original_img_path)

                # Correct label file if needed
                label_file = base_name + ".txt"
                label_path = os.path.join(src_cls_val_labels, label_file)

                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        lines = f.readlines()

                    new_lines = []
                    needs_correction = False
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) > 1:
                            if parts[0] != str(cls_id):
                                needs_correction = True
                                parts[0] = str(cls_id)
                            new_lines.append(' '.join(parts) + '\n')

                    if needs_correction:
                        with open(label_path, 'w') as f:
                            f.writelines(new_lines)
                        labels_corrected += 1

            total_val += len(val_files)
            print(f"  {cls_name}: {len(val_files)} images")

    # Write train.txt
    train_txt_path = os.path.join(subset_dir, "train.txt")
    with open(train_txt_path, 'w') as f:
        for img_path in train_image_paths:
            f.write(img_path + '\n')

    # Write val.txt
    val_txt_path = os.path.join(subset_dir, "val.txt")
    with open(val_txt_path, 'w') as f:
        for img_path in val_image_paths:
            f.write(img_path + '\n')

    # Create data.yaml
    data_yaml_path = os.path.join(subset_dir, "data.yaml")
    data_config = {
        'path': data_root,
        'train': train_txt_path,
        'val': val_txt_path,
        'nc': len(classes),
        'names': {v: k for k, v in classes.items()}
    }

    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)

    print(f"\n{'='*50}")
    print(f"Sampled dataset created:")
    print(f"  Training images: {total_train} (from Train/)")
    print(f"  Validation images: {total_val} (from val/)")
    print(f"  Labels corrected: {labels_corrected}")
    print(f"  Config: {data_yaml_path}")
    print(f"{'='*50}")

    return data_yaml_path


def train_model(model_source, model_name, data_yaml, training_config, base_dir,
                use_full_dataset=True, checkpoint_dir=None, default_warmup_epochs=3):
    """
    Train the RT-DETR model and return training results.

    For full dataset training:
      - Checks for existing checkpoint and resumes from it
      - Tracks total epochs across multiple training sessions
      - Saves checkpoint after training completes
      - Uses single fixed folder per model (no timestamps)

    For sampled dataset training:
      - Always starts fresh (no resume)
      - Uses single fixed folder per model (no timestamps)

    Args:
        model_source: Path to model YAML config or pretrained .pt file
        model_name: Name of the model (used for folder naming)
        data_yaml: Path to data configuration YAML
        training_config: Dict with training hyperparameters
        base_dir: Base output directory
        use_full_dataset: Whether using full dataset (enables checkpointing)
        checkpoint_dir: Directory for checkpoints (required if use_full_dataset=True)
        default_warmup_epochs: Default warmup epochs (3 for from-scratch, 1 for pretrained)

    Returns:
        Dict with training results including paths, metrics, and resume info
    """
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")

    # Project directory
    project_dir = os.path.join(base_dir, "training_runs")
    os.makedirs(project_dir, exist_ok=True)

    # Fixed run name (no timestamp) - one folder per model
    run_name = model_name
    run_dir = os.path.join(project_dir, run_name)

    # Initialize resume tracking
    resume_from_checkpoint = False
    previous_epochs = 0
    checkpoint_model_path = None
    checkpoint_meta_path = None

    if use_full_dataset and checkpoint_dir is not None:
        checkpoint_model_path = os.path.join(checkpoint_dir, "last.pt")
        checkpoint_meta_path = os.path.join(checkpoint_dir, "training_meta.json")

        # Check for existing checkpoint
        if os.path.exists(checkpoint_model_path) and os.path.exists(checkpoint_meta_path):
            with open(checkpoint_meta_path, 'r') as f:
                meta = json.load(f)
            previous_epochs = meta.get('total_epochs', 0)
            resume_from_checkpoint = True
            print(f"\n*** RESUMING FROM CHECKPOINT ***")
            print(f"  Previous epochs completed: {previous_epochs}")
            print(f"  This session will train epochs: {previous_epochs + 1} to {previous_epochs + training_config['epochs']}")
            print(f"  Loading model from: {checkpoint_model_path}")
        else:
            print(f"\nNo checkpoint found. Starting fresh training.")
            # Clear existing run folder for fresh start
            if os.path.exists(run_dir):
                print(f"Clearing previous run folder: {run_name}")
                shutil.rmtree(run_dir)
    else:
        # Sampled dataset - always start fresh
        print(f"\nSampled dataset mode - starting fresh training.")
        # Clear existing run folder
        if os.path.exists(run_dir):
            print(f"Clearing previous run folder: {run_name}")
            shutil.rmtree(run_dir)

    # Load model - either from checkpoint or from source (YAML or .pt)
    if resume_from_checkpoint:
        model = RTDETR(checkpoint_model_path)
    else:
        model = RTDETR(model_source)

    # Record start time
    start_time = time.time()

    # Train - use exist_ok=True to allow updating existing folder
    results = model.train(
        data=data_yaml,
        epochs=training_config["epochs"],
        imgsz=training_config["imgsz"],
        batch=training_config["batch"],
        lr0=training_config["lr0"],
        lrf=training_config["lrf"],
        momentum=training_config["momentum"],
        weight_decay=training_config["weight_decay"],
        workers=training_config["workers"],
        patience=training_config["patience"],
        cos_lr=training_config["cos_lr"],
        warmup_epochs=training_config.get("warmup_epochs", default_warmup_epochs) if not resume_from_checkpoint else 0,
        warmup_momentum=training_config.get("warmup_momentum", 0.8),
        warmup_bias_lr=training_config.get("warmup_bias_lr", 0.1),
        project=project_dir,
        name=run_name,
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True,
    )

    training_time = time.time() - start_time

    # Get model paths from this run
    best_model_path = os.path.join(run_dir, "weights", "best.pt")
    last_model_path = os.path.join(run_dir, "weights", "last.pt")

    # Calculate total epochs
    total_epochs = previous_epochs + training_config["epochs"]

    # Save checkpoint for full dataset training
    if use_full_dataset and checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Copy last.pt to checkpoint directory
        if os.path.exists(last_model_path):
            shutil.copy2(last_model_path, checkpoint_model_path)
            print(f"\nCheckpoint saved: {checkpoint_model_path}")

        # Save/update metadata
        meta = {
            'total_epochs': total_epochs,
            'last_run_epochs': training_config["epochs"],
            'last_run_dir': run_dir,
            'last_run_time': datetime.now().isoformat(),
            'training_sessions': 1,
        }

        # Load existing meta to preserve history
        if resume_from_checkpoint:
            with open(checkpoint_meta_path, 'r') as f:
                old_meta = json.load(f)
            meta['training_sessions'] = old_meta.get('training_sessions', 0) + 1
            meta['history'] = old_meta.get('history', [])
            meta['history'].append({
                'session': meta['training_sessions'],
                'epochs_this_session': training_config["epochs"],
                'total_epochs_after': total_epochs,
                'run_dir': run_dir,
                'timestamp': datetime.now().isoformat(),
            })
        else:
            meta['history'] = [{
                'session': 1,
                'epochs_this_session': training_config["epochs"],
                'total_epochs_after': total_epochs,
                'run_dir': run_dir,
                'timestamp': datetime.now().isoformat(),
            }]

        with open(checkpoint_meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"Metadata saved: {checkpoint_meta_path}")

        print(f"\n*** TOTAL EPOCHS COMPLETED: {total_epochs} ***")

    return {
        "model_name": model_name,
        "best_model_path": best_model_path,
        "training_time": training_time,
        "run_dir": run_dir,
        "results": results,
        "total_epochs": total_epochs,
        "previous_epochs": previous_epochs,
        "resumed": resume_from_checkpoint,
    }


def evaluate_model(model_path, images_dir, classes, id2label,
                   conf_thresh=0.1, eval_per_class=100, random_seed=123):
    """
    Evaluate the trained model on the dataset.

    Args:
        model_path: Path to the trained model weights
        images_dir: Directory containing class subdirectories with images
        classes: Dict mapping class names to class IDs
        id2label: Dict mapping class IDs to class names
        conf_thresh: Confidence threshold for predictions
        eval_per_class: Maximum number of samples to evaluate per class
        random_seed: Random seed for reproducibility

    Returns:
        Dict with evaluation metrics including accuracy, confusion matrix,
        classification report, and inference times
    """
    model = RTDETR(model_path)

    random.seed(random_seed)

    y_true = []
    y_pred = []
    inference_times = []

    for gt_class, gt_id in classes.items():
        cls_dir = os.path.join(images_dir, gt_class)
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(".jpg")]

        if len(files) > eval_per_class:
            files = random.sample(files, eval_per_class)

        for fname in tqdm(files, desc=f"Evaluating {gt_class}", leave=False):
            img_path = os.path.join(cls_dir, fname)

            start = time.time()
            results = model(img_path, conf=conf_thresh, verbose=False)[0]
            inference_times.append(time.time() - start)

            y_true.append(gt_id)

            if len(results.boxes) == 0:
                y_pred.append(-1)
            else:
                best_idx = results.boxes.conf.argmax()
                y_pred.append(int(results.boxes.cls[best_idx].cpu().item()))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate metrics
    valid = y_pred != -1
    valid_count = np.sum(valid)

    if valid_count > 0:
        accuracy = accuracy_score(y_true[valid], y_pred[valid])
        cm = confusion_matrix(y_true[valid], y_pred[valid], labels=list(range(len(classes))))
        report = classification_report(
            y_true[valid], y_pred[valid],
            target_names=list(classes.keys()),
            labels=list(range(len(classes))),
            zero_division=0,
            output_dict=True
        )
    else:
        accuracy = 0.0
        cm = None
        report = None

    return {
        "accuracy": accuracy,
        "no_prediction_count": len(y_true) - valid_count,
        "total_samples": len(y_true),
        "confusion_matrix": cm.tolist() if cm is not None else None,
        "classification_report": report,
        "avg_inference_time": np.mean(inference_times),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


def convert_to_native(obj):
    """
    Convert numpy types to native Python types for JSON serialization.

    Args:
        obj: Object to convert (can be numpy array, dict, list, or scalar)

    Returns:
        Object with all numpy types converted to native Python types
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(i) for i in obj]
    return obj


def create_full_dataset_config(data_root, base_dir, num_classes, id2label):
    """
    Create data configuration YAML for full dataset training.

    Args:
        data_root: Root directory containing Train/ and val/ folders
        base_dir: Output directory for configuration file
        num_classes: Number of classes
        id2label: Dict mapping class IDs to class names

    Returns:
        Path to the generated data.yaml configuration file
    """
    data_yaml_path = os.path.join(base_dir, "data_full.yaml")
    data_config = {
        'path': data_root,
        'train': 'Train/images',
        'val': 'val/images',
        'nc': num_classes,
        'names': id2label
    }

    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)

    return data_yaml_path


def save_results(results_dir, model_name, backbone, is_pretrained,
                 training_result, evaluation_result, training_config, classes):
    """
    Save training and evaluation results to JSON file.

    Args:
        results_dir: Directory to save results
        model_name: Name of the model
        backbone: Backbone architecture name
        is_pretrained: Whether model was pretrained
        training_result: Dict from train_model()
        evaluation_result: Dict from evaluate_model()
        training_config: Training hyperparameters dict
        classes: Dict mapping class names to class IDs

    Returns:
        Path to the saved results JSON file
    """
    results_to_save = {
        "model_name": model_name,
        "backbone": backbone,
        "is_pretrained": is_pretrained,
        "best_model_path": training_result["best_model_path"],
        "run_dir": training_result["run_dir"],
        "training_time_s": float(training_result["training_time"]),
        "training_config": training_config,
        "total_epochs": training_result.get("total_epochs", training_config["epochs"]),
        "resumed_training": training_result.get("resumed", False),
        "accuracy": float(evaluation_result["accuracy"]),
        "avg_inference_time_ms": float(evaluation_result["avg_inference_time"]) * 1000,
        "no_prediction_count": int(evaluation_result["no_prediction_count"]),
        "total_samples": int(evaluation_result["total_samples"]),
        "confusion_matrix": convert_to_native(evaluation_result["confusion_matrix"]),
        "classification_report": convert_to_native(evaluation_result["classification_report"]),
        "y_true": convert_to_native(evaluation_result["y_true"]),
        "y_pred": convert_to_native(evaluation_result["y_pred"]),
        "classes": classes,
        "timestamp": datetime.now().isoformat(),
    }

    results_file = os.path.join(results_dir, f"{model_name}_results.json")
    with open(results_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)

    return results_file


def print_training_summary(model_name, backbone, training_result, evaluation_result,
                           training_config, checkpoint_model_path=None, results_file=None):
    """
    Print a summary of training results.

    Args:
        model_name: Name of the model
        backbone: Backbone architecture name
        training_result: Dict from train_model()
        evaluation_result: Dict from evaluate_model()
        training_config: Training hyperparameters dict
        checkpoint_model_path: Path to checkpoint (if using full dataset)
        results_file: Path to saved results JSON
    """
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Model: {model_name} ({backbone})")
    print(f"Total Epochs: {training_result.get('total_epochs', training_config['epochs'])}")
    if training_result.get('resumed', False):
        print(f"  (Resumed from epoch {training_result['previous_epochs'] + 1})")
    print(f"Accuracy: {evaluation_result['accuracy']:.4f}")
    print(f"Inference Time: {evaluation_result['avg_inference_time']*1000:.2f}ms")
    print(f"Training Time (this session): {training_result['training_time']:.1f}s")
    print(f"\nBest model: {training_result['best_model_path']}")
    if checkpoint_model_path:
        print(f"Checkpoint: {checkpoint_model_path}")
    if results_file:
        print(f"Results JSON: {results_file}")
    print("="*60)
