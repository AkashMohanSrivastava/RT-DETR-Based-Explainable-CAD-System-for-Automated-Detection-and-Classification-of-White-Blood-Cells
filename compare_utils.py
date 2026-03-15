"""
compare_utils.py
────────────────────────────────────────────────────────────────────────────────
Utility functions and classes for Compare_and_Visualize_Results.ipynb.

Sections
--------
1. Qualitative Label Helpers
2. Results Loading
3. Loss Curve Loading
4. Visualisation Helpers
5. Analysis and Rankings
6. Explainability — GradCAM and ScoreCAM
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
)
from ultralytics import RTDETR


# ── Loss column names expected in Ultralytics results.csv ─────────────────────
TRAIN_LOSS_COLS = ["train/giou_loss", "train/cls_loss", "train/l1_loss"]
VAL_LOSS_COLS   = ["val/giou_loss",   "val/cls_loss",   "val/l1_loss"]


# ══════════════════════════════════════════════════════════════════════════════
# 1. Qualitative Label Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _kappa_label(k):
    """Return a human-readable quality label for a Cohen's Kappa value.

    Thresholds follow the Landis & Koch (1977) convention:
        > 0.80 Excellent · > 0.60 Good · > 0.40 Moderate
        > 0.20 Fair      · else   Poor
    """
    if k is None: return "N/A"
    if k > 0.80:  return "Excellent"
    if k > 0.60:  return "Good"
    if k > 0.40:  return "Moderate"
    if k > 0.20:  return "Fair"
    return "Poor"


def _mcc_label(m):
    """Return a human-readable quality label for a Matthews Correlation Coefficient.

    Thresholds:
        > 0.70 Strong · > 0.50 Good · > 0.30 Moderate · else Weak
    """
    if m is None: return "N/A"
    if m > 0.70:  return "Strong"
    if m > 0.50:  return "Good"
    if m > 0.30:  return "Moderate"
    return "Weak"


def _size_note(s):
    """Return a one-line deployment suitability comment for a model size in MB."""
    if s is None: return ""
    if s < 20:    return "  -> Very lightweight; ideal for edge/mobile deployment"
    if s < 60:    return "  -> Compact; suitable for most server deployments"
    if s < 150:   return "  -> Medium-sized; standard GPU deployment"
    return               "  -> Large model; may need optimisation (quantisation/pruning) for edge use"


def _params_note(p):
    """Return a one-line capacity comment for a parameter count in millions."""
    if p is None: return ""
    if p < 5:     return "  -> Very few parameters; low capacity, fast inference"
    if p < 20:    return "  -> Lightweight backbone; good accuracy/speed trade-off"
    if p < 50:    return "  -> Mid-range; typical for production detectors"
    return               "  -> High capacity; ensure accuracy gain justifies cost"


# ══════════════════════════════════════════════════════════════════════════════
# 2. Results Loading
# ══════════════════════════════════════════════════════════════════════════════

def _load_map_from_csv(result, base_dir):
    """Read best mAP50 and mAP50-95 from an Ultralytics results CSV.

    Search order
    ────────────
    1. ``{base_dir}/checkpoints/<model>_full_results.csv``
    2. ``result["run_dir"]/results.csv``  (if the key is present)

    Parameters
    ----------
    result   : dict   Single model result dict (must contain ``model_name``).
    base_dir : str    Project base directory used to locate checkpoints.

    Returns
    -------
    (mAP50, mAP50_95) : tuple[float | None, float | None]
    """
    MAP50_COL   = "metrics/mAP50(B)"
    MAP5095_COL = "metrics/mAP50-95(B)"

    candidates = [
        os.path.join(base_dir, "checkpoints",
                     f"{result['model_name']}_full_results.csv"),
    ]
    run_dir = result.get("run_dir", "")
    if run_dir:
        candidates.append(os.path.join(run_dir, "results.csv"))

    for csv_path in candidates:
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip()
                if MAP50_COL in df.columns:
                    mAP50   = float(df[MAP50_COL].max())
                    mAP5095 = (float(df[MAP5095_COL].max())
                               if MAP5095_COL in df.columns else None)
                    return mAP50, mAP5095
            except Exception:
                pass
    return None, None


def _compute_extra_metrics(result):
    """Compute MCC, Cohen's Kappa, and Balanced Accuracy from stored predictions.

    Reads ``y_true`` and ``y_pred`` from the result dict.  Samples where
    ``y_pred == -1`` (no detection) are excluded before computing metrics.

    Returns
    -------
    (mcc, kappa, balanced_acc) : tuple[float | None, float | None, float | None]
    """
    y_true = np.array(result.get("y_true", []))
    y_pred = np.array(result.get("y_pred", []))
    if len(y_true) == 0 or len(y_pred) == 0:
        return None, None, None

    valid    = y_pred != -1
    y_true_v = y_true[valid]
    y_pred_v = y_pred[valid]
    if len(y_true_v) == 0:
        return None, None, None

    try:
        mcc     = float(matthews_corrcoef(y_true_v, y_pred_v))
        kappa   = float(cohen_kappa_score(y_true_v, y_pred_v))
        bal_acc = float(balanced_accuracy_score(y_true_v, y_pred_v))
        return mcc, kappa, bal_acc
    except Exception:
        return None, None, None


def _get_model_size_and_params(best_model_path):
    """Return (size_mb, params_m) by inspecting a checkpoint file.

    Avoids full model initialisation — loads only the state-dict to count
    tensor elements.

    Returns
    -------
    (size_mb, params_m) : tuple[float | None, float | None]
    """
    if not best_model_path or not os.path.exists(best_model_path):
        return None, None

    size_mb = os.path.getsize(best_model_path) / (1024 ** 2)
    try:
        ckpt = torch.load(best_model_path, map_location="cpu")
        sd   = ckpt.get("model", ckpt)
        if hasattr(sd, "state_dict"):
            sd = sd.state_dict()
        n_params = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
        params_m = n_params / 1e6
    except Exception:
        params_m = None

    return size_mb, params_m


def load_all_results(results_dir, model_names, base_dir):
    """Load and enrich result JSON files for all trained models.

    For each model, the JSON is augmented with:
    - mAP50 / mAP50-95 (from Ultralytics results CSV)
    - MCC, Cohen's Kappa, Balanced Accuracy (from stored predictions)
    - Model file size (MB) and parameter count (M)

    Parameters
    ----------
    results_dir  : str        Directory containing ``<model_name>_results.json`` files.
    model_names  : list[str]  Ordered list of model names to load.
    base_dir     : str        Project base directory (used to locate checkpoint CSVs).

    Returns
    -------
    list[dict]
        Enriched result dicts, one per successfully loaded model.
    """
    all_results    = []
    missing_models = []

    for model_name in model_names:
        results_file = os.path.join(results_dir, f"{model_name}_results.json")

        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                result = json.load(f)

            mAP50, mAP5095           = _load_map_from_csv(result, base_dir)
            result["mAP50"]          = mAP50
            result["mAP50_95"]       = mAP5095

            mcc, kappa, bal_acc      = _compute_extra_metrics(result)
            result["mcc"]            = mcc
            result["cohen_kappa"]    = kappa
            result["balanced_accuracy"] = bal_acc

            size_mb, params_m        = _get_model_size_and_params(
                                           result.get("best_model_path", ""))
            result["model_size_mb"]  = size_mb
            result["params_m"]       = params_m

            all_results.append(result)
            epochs  = result.get("total_epochs", "N/A")
            map_str = f", mAP50: {mAP50:.4f}" if mAP50 is not None else ""
            mcc_str = f", MCC: {mcc:.4f}"     if mcc  is not None else ""
            print(f"Loaded: {model_name} (Accuracy: {result['accuracy']:.4f}, "
                  f"Epochs: {epochs}{map_str}{mcc_str})")
        else:
            missing_models.append(model_name)
            print(f"WARNING: Results not found for {model_name}")

    if missing_models:
        print(f"\nMissing models: {missing_models}")
        print("Please run the corresponding training notebooks first.")

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# 3. Loss Curve Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_loss_curves(run_dir, ckpt_dir, model_name):
    """Load training and validation loss curves for a single model.

    Prefers ``{ckpt_dir}/full_results.csv`` (accumulated across all training
    sessions) over the session-only ``{run_dir}/results.csv``.  When the full
    CSV is present epoch numbers are already cumulative and no offset is applied.

    Parameters
    ----------
    run_dir    : str   Path to the Ultralytics training run directory.
    ckpt_dir   : str   Path to the checkpoint directory for this model.
    model_name : str   Model name used in warning messages.

    Returns
    -------
    pd.DataFrame or None
        Columns: ``epoch_cumulative``, ``train_loss``, ``val_loss``,
        ``val_ok``, ``total_epochs``, ``source``.
        Returns ``None`` if no CSV is found.
    """
    full_csv  = os.path.join(ckpt_dir, "full_results.csv")
    sess_csv  = os.path.join(run_dir,  "results.csv")
    meta_path = os.path.join(ckpt_dir, "training_meta.json")

    use_full = os.path.exists(full_csv)
    csv_path = full_csv if use_full else sess_csv

    if not os.path.exists(csv_path):
        print(f"  WARNING: No loss CSV found for {model_name}")
        return None

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    df["train_loss"] = df[TRAIN_LOSS_COLS].sum(axis=1)
    val_has_data     = not df[VAL_LOSS_COLS].isnull().all().all()
    df["val_loss"]   = df[VAL_LOSS_COLS].sum(axis=1) if val_has_data else np.nan
    df["val_ok"]     = val_has_data

    epoch_offset = 0
    total_epochs = len(df)
    if not use_full and os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)
        total_epochs = meta.get("total_epochs",    len(df))
        last_run_ep  = meta.get("last_run_epochs", len(df))
        epoch_offset = total_epochs - last_run_ep
    elif use_full and os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)
        total_epochs = meta.get("total_epochs", len(df))

    df["epoch_cumulative"] = df["epoch"] + epoch_offset
    df["total_epochs"]     = total_epochs
    df["source"]           = ("full_results.csv (all sessions)" if use_full
                               else "results.csv (last session only)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. Visualisation Helpers
# ══════════════════════════════════════════════════════════════════════════════

def plot_bar(ax, names, values, ylabel, title, fmt=".3f", ylim=None, baseline=None):
    """Draw a colour-coded, labelled bar chart on *ax*.

    Parameters
    ----------
    ax       : matplotlib.axes.Axes
    names    : list[str]              Bar labels (model names).
    values   : list[float | None]     Bar heights; ``None`` entries render as
                                      zero with an "N/A" annotation.
    ylabel   : str                    Y-axis label.
    title    : str                    Chart title.
    fmt      : str                    Python format spec for value labels
                                      (default ``".3f"``).
    ylim     : tuple[float,float] | None   Y-axis limits.
    baseline : float | None           If given, draws a dashed red reference line.
    """
    import matplotlib.pyplot as plt  # deferred to avoid mandatory pyplot import

    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    safe   = [v if v is not None else 0.0 for v in values]
    ax.bar(names, safe, color=colors)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=12)
    ax.tick_params(axis="y", labelsize=11)
    if ylim:
        ax.set_ylim(*ylim)
    if baseline is not None:
        ax.axhline(baseline, color="red", linestyle="--", linewidth=1,
                   label=f"Random baseline ({baseline})")
        ax.legend(fontsize=11)
    ymax = max(safe) if max(safe) > 0 else 1
    for i, v in enumerate(safe):
        label = f"{v:{fmt}}" if values[i] is not None else "N/A"
        ax.text(i, v + ymax * 0.02, label, ha="center", fontsize=11, fontweight="bold")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Analysis and Rankings
# ══════════════════════════════════════════════════════════════════════════════

def best_result(results, key):
    """Return the result dict with the highest value for *key*, ignoring ``None``.

    Parameters
    ----------
    results : list[dict]
    key     : str   Metric key to maximise.

    Returns
    -------
    dict or None
    """
    valid = [r for r in results if r.get(key) is not None]
    return max(valid, key=lambda x: x[key]) if valid else None


def rank_results(results, key, reverse=True):
    """Return results sorted by *key*, excluding entries where the key is ``None``.

    Parameters
    ----------
    results : list[dict]
    key     : str   Metric key to sort by.
    reverse : bool  ``True`` = descending (higher is better).  Default ``True``.

    Returns
    -------
    list[dict]
    """
    valid = [r for r in results if r.get(key) is not None]
    return sorted(valid, key=lambda x: x[key], reverse=reverse)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Explainability — GradCAM and ScoreCAM
# ══════════════════════════════════════════════════════════════════════════════

def pick_best_samples(model_path, images_dir, classes, n_candidates=20, seed=42):
    """Select the highest-confidence correctly-predicted image per class.

    Loads *model_path* once, samples up to *n_candidates* images from
    ``images_dir/<class_name>/`` for each class, runs a fast forward pass
    (no GradCAM overhead), and returns the image with the highest prediction
    confidence where the predicted class matches the true class.  Falls back
    to a random candidate if no correct prediction is found in the sample.

    Parameters
    ----------
    model_path   : str    Path to a trained RT-DETR ``.pt`` checkpoint.
    images_dir   : str    Root directory; sub-directories named by class.
    classes      : dict   ``{class_name: class_id}`` mapping.
    n_candidates : int    Images to sample per class (default 20).
    seed         : int    Random seed for reproducible sampling (default 42).

    Returns
    -------
    dict  ``{class_name: img_path}``
    """
    import random
    rng = random.Random(seed)

    cam_model     = ExplainableRTDETR(model_path)
    sample_images = {}

    for cls_name, cls_id in classes.items():
        cls_dir    = os.path.join(images_dir, cls_name)
        all_files  = [f for f in os.listdir(cls_dir) if f.lower().endswith(".jpg")]
        candidates = rng.sample(all_files, min(n_candidates, len(all_files)))

        best_path, best_conf = None, -1.0
        fallback = os.path.join(cls_dir, candidates[0])

        for fname in candidates:
            img_path = os.path.join(cls_dir, fname)
            try:
                conf, pred_cls = cam_model._predict_conf(img_path)
                if pred_cls == cls_id and conf > best_conf:
                    best_conf, best_path = conf, img_path
            except Exception:
                continue

        sample_images[cls_name] = best_path if best_path else fallback
        status = f"conf={best_conf:.3f}" if best_path else "fallback (no correct pred in sample)"
        print(f"  {cls_name}: {os.path.basename(sample_images[cls_name])}  [{status}]")

    cam_model.cleanup()
    return sample_images


class ExplainableRTDETR:
    """GradCAM and ScoreCAM visualisation for Ultralytics RT-DETR models.

    Hooks into the last backbone feature layer (before the AIFI encoder) to
    produce class-discriminative heatmaps that highlight which image regions
    most influenced the model's classification decision.

    Parameters
    ----------
    model_path : str   Path to a trained RT-DETR ``.pt`` checkpoint.
    imgsz      : int   Input resolution (square). Default ``512``.

    Examples
    --------
    >>> explainer = ExplainableRTDETR("best.pt")
    >>> overlay, cam, orig, pred_cls = explainer.gradcam("image.jpg")
    >>> overlay, cam, orig, pred_cls = explainer.scorecam("image.jpg")
    >>> explainer.cleanup()
    """

    def __init__(self, model_path, imgsz=512):
        self.yolo      = RTDETR(model_path)
        self.det_model = self.yolo.model
        self.device    = next(self.det_model.parameters()).device
        self.det_model.eval()
        self.imgsz = imgsz

        self.activations = None
        self.gradients   = None

        self.target_layer, self.target_idx = self._find_target_layer()
        self._handles = self._register_hooks()
        print(f"  Target layer: index {self.target_idx}  "
              f"({type(self.target_layer).__name__})")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _find_target_layer(self):
        """Return the last C2f / C3 / SPPF backbone layer before AIFI.

        Falls back to the last Conv layer, then the first module, if none
        of the preferred types are found.
        """
        target, idx           = None, -1
        conv_target, conv_idx = None, -1
        for i, m in enumerate(self.det_model.model):
            if type(m).__name__ == "AIFI":
                break
            name = type(m).__name__
            if name in ("C2f", "C3", "SPPF", "Bottleneck"):
                target, idx = m, i
            elif name == "Conv":
                conv_target, conv_idx = m, i
        if target      is not None: return target,      idx
        if conv_target is not None: return conv_target, conv_idx
        for i, m in enumerate(self.det_model.model):
            if type(m).__name__ == "Conv":
                return m, i
        return self.det_model.model[0], 0

    def _find_all_backbone_layers(self):
        """Return ``[(idx, module)]`` for every feature block before AIFI.

        Prefers C2f / C3 / SPPF / Bottleneck; falls back to Conv layers.
        """
        preferred, convs = [], []
        for i, m in enumerate(self.det_model.model):
            if type(m).__name__ == "AIFI":
                break
            name = type(m).__name__
            if name in ("C2f", "C3", "SPPF", "Bottleneck"):
                preferred.append((i, m))
            elif name == "Conv":
                convs.append((i, m))
        return preferred if preferred else convs

    def _register_hooks(self):
        h1 = self.target_layer.register_forward_hook(
            lambda m, inp, out: setattr(
                self, "activations",
                out if isinstance(out, torch.Tensor) else out[0],
            )
        )
        h2 = self.target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "gradients", go[0])
        )
        return [h1, h2]

    def _preprocess(self, img_path):
        """Read image → ``(1, 3, H, W)`` float32 tensor in [0, 1] + original RGB."""
        img_bgr     = cv2.imread(img_path)
        img_rgb     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.imgsz, self.imgsz))
        tensor = (
            torch.from_numpy(img_resized)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(self.device)
        )
        return tensor, img_rgb

    def _extract_target_score(self, output):
        """Parse decoder output and return ``(score_tensor, class_id_int)``."""
        scores = output[0] if isinstance(output, (tuple, list)) else output
        while isinstance(scores, (tuple, list)):
            scores = scores[0]
        if scores.dim() == 3:
            scores = scores[0]
        if scores.shape[-1] > 5:
            scores = scores[:, 4:]
        best_q = scores.max(dim=1).values.argmax()
        best_c = scores[best_q].argmax()
        return scores[best_q, best_c], int(best_c.item())

    @staticmethod
    def _to_heatmap(cam, orig_img):
        """Resize, normalise, and overlay a raw CAM onto the original image.

        Returns
        -------
        overlay  : np.ndarray (H, W, 3) uint8   Blended heatmap + image.
        cam_norm : np.ndarray (H, W)             Normalised CAM in [0, 1].
        """
        h, w = orig_img.shape[:2]
        cam  = cv2.resize(cam.astype(np.float32), (w, h))
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = np.uint8(np.clip(
            heatmap.astype(np.float32) * 0.4
            + orig_img.astype(np.float32) * 0.6,
            0, 255,
        ))
        return overlay, cam

    # ── Public API ────────────────────────────────────────────────────────────

    def gradcam(self, img_path):
        """Compute multi-layer GradCAM for *img_path*.

        Aggregates gradient-weighted activation maps from every C2f / SPPF /
        Bottleneck backbone layer before AIFI:

            weights   = global-average-pool(gradients)
            layer_cam = ReLU(Σ_c  weights_c × activations_c)

        Each per-layer CAM is normalised independently, upsampled to input
        resolution, and summed to form the final heatmap.

        Returns
        -------
        overlay  : np.ndarray (H, W, 3) uint8   Heatmap blended with original.
        cam_norm : np.ndarray (H, W)             Normalised CAM in [0, 1].
        orig_img : np.ndarray (H, W, 3)          Original RGB image.
        pred_cls : int                            Predicted class index.
        """
        tensor, orig_img = self._preprocess(img_path)
        h_in, w_in = tensor.shape[2], tensor.shape[3]
        tensor.requires_grad_(True)

        selected = self._find_all_backbone_layers()
        if not selected:
            selected = [(self.target_idx, self.target_layer)]

        layer_acts, layer_grads, tmp_handles = {}, {}, []
        for layer_idx, mod in selected:
            def _fwd(m, inp, out, i=layer_idx):
                layer_acts[i] = out if isinstance(out, torch.Tensor) else out[0]
            def _bwd(m, gi, go, i=layer_idx):
                layer_grads[i] = go[0]
            tmp_handles.append(mod.register_forward_hook(_fwd))
            tmp_handles.append(mod.register_full_backward_hook(_bwd))

        self.det_model.zero_grad()
        with torch.enable_grad():
            output = self.det_model(tensor)

        score, pred_cls = self._extract_target_score(output)
        score.backward()

        for h in tmp_handles:
            h.remove()

        if not layer_acts or not layer_grads:
            return None, None, orig_img, pred_cls

        cam_final = np.zeros((h_in, w_in), dtype=np.float32)
        for layer_idx in layer_acts:
            if layer_idx not in layer_grads:
                continue
            acts  = layer_acts[layer_idx].detach()
            grads = layer_grads[layer_idx].detach()
            weights   = grads.mean(dim=[2, 3], keepdim=True)
            layer_cam = F.relu((weights * acts).sum(dim=1))
            layer_cam = layer_cam.squeeze().cpu().numpy()
            layer_cam = np.maximum(layer_cam, 0)
            if layer_cam.max() > 0:
                layer_cam /= layer_cam.max()
            cam_final += cv2.resize(layer_cam, (w_in, h_in))

        cam_final = np.maximum(cam_final, 0)
        overlay, cam_norm = self._to_heatmap(cam_final, orig_img)
        return overlay, cam_norm, orig_img, pred_cls

    def scorecam(self, img_path, top_k=16):
        """Compute ScoreCAM (gradient-free) for *img_path*.

        Uses the single ``target_layer``.  Each of the top-k activation maps
        (ranked by mean activation) is upsampled into a soft input mask; the
        resulting class confidence score weights that map in the final sum.

        Parameters
        ----------
        img_path : str
        top_k    : int   Number of activation channels to use (default 16).

        Returns
        -------
        Same four-element tuple as :meth:`gradcam`.
        """
        tensor, orig_img = self._preprocess(img_path)
        self.activations = None

        with torch.no_grad():
            output = self.det_model(tensor)

        if self.activations is None:
            return None, None, orig_img, -1

        _, pred_cls = self._extract_target_score(output)
        acts        = self.activations.detach()

        mean_per_ch = acts[0].mean(dim=[1, 2])
        k           = min(top_k, acts.shape[1])
        topk_idx    = mean_per_ch.topk(k).indices
        h_in, w_in  = tensor.shape[2], tensor.shape[3]

        weights = []
        for idx in topk_idx:
            act_map      = acts[0, idx].unsqueeze(0).unsqueeze(0)
            mask         = F.interpolate(act_map, size=(h_in, w_in),
                                         mode="bilinear", align_corners=False)
            mask         = mask.squeeze()
            mask         = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
            masked_input = tensor * mask.unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                masked_out = self.det_model(masked_input)
            sc, _ = self._extract_target_score(masked_out)
            weights.append(max(sc.item(), 0.0))

        weights = np.array(weights, dtype=np.float32)
        cam     = np.zeros((acts.shape[2], acts.shape[3]), dtype=np.float32)
        for i, idx in enumerate(topk_idx):
            cam += weights[i] * acts[0, idx.item()].cpu().numpy()
        cam = np.maximum(cam, 0)

        overlay, cam_norm = self._to_heatmap(cam, orig_img)
        return overlay, cam_norm, orig_img, pred_cls

    def _predict_conf(self, img_path):
        """Fast forward-pass — returns ``(confidence, pred_class_int)`` with no CAM overhead."""
        tensor, _ = self._preprocess(img_path)
        with torch.no_grad():
            output = self.det_model(tensor)
        score, cls_id = self._extract_target_score(output)
        return float(score.item()), cls_id

    def cleanup(self):
        """Remove all registered forward/backward hooks."""
        for h in self._handles:
            h.remove()
