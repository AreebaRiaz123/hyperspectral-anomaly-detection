"""Helper functions: data loading, normalization, evaluation and plotting."""
import os
import urllib.request

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score, roc_curve

DATA_URL = "https://raw.githubusercontent.com/jeline0110/GT-HAD/main/data/{}.mat"


def download_dataset(name, folder="data"):
    """Download a benchmark .mat file if it is not already present."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.mat")
    if not os.path.exists(path):
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(DATA_URL.format(name), path)
    return path


def load_hsi(path):
    """Load a hyperspectral cube (H x W x B) and its ground-truth map (H x W)."""
    mat = loadmat(path)
    data = mat["data"].astype(np.float64)
    gt = (mat["map"] > 0).astype(int)
    return data, gt


def normalize(data):
    """Min-max normalize the whole cube to [0, 1]."""
    return (data - data.min()) / (data.max() - data.min())


def evaluate(score_map, gt):
    """Return AUC and ROC curve (FPR, TPR) for a detection map."""
    y_true, y_score = gt.ravel(), score_map.ravel()
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc, fpr, tpr


def false_color(data):
    """Build a simple false-colour RGB image from three bands."""
    B = data.shape[2]
    rgb = data[:, :, [int(B * 0.75), int(B * 0.5), int(B * 0.25)]]
    return (rgb - rgb.min()) / (rgb.max() - rgb.min())


def plot_detection_maps(name, data, gt, maps, aucs, save_path):
    """Save image, ground truth and every method's detection map in a grid."""
    panels = [("False colour", false_color(data), None), ("Ground truth", gt, "gray")]
    panels += [(f"{m}\nAUC={aucs[m]:.4f}", v, "jet") for m, v in maps.items()]
    cols = 4
    rows = int(np.ceil(len(panels) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    ax = np.atleast_1d(ax).ravel()
    for a in ax:
        a.axis("off")
    for a, (title, img, cmap) in zip(ax, panels):
        a.imshow(img, cmap=cmap)
        a.set_title(title)
    fig.suptitle(name, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=110)
    plt.close()


def plot_roc(name, rocs, aucs, save_path):
    """Save ROC curves of all methods (log-scale false alarm rate)."""
    plt.figure(figsize=(6, 5))
    for method, (fpr, tpr) in rocs.items():
        plt.plot(fpr, tpr, label=f"{method} ({aucs[method]:.4f})")
    plt.xscale("log"); plt.xlim(1e-4, 1)
    plt.xlabel("False alarm rate"); plt.ylabel("Detection rate")
    plt.title(f"ROC curves - {name}"); plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=110)
    plt.close()


def plot_separability(name, maps, gt, save_path):
    """Box plot of normalized scores for background vs anomaly pixels.

    A good detector has a large gap between the two boxes: anomalies high,
    background low and compact.
    """
    data, labels, colors = [], [], []
    for method, m in maps.items():
        s = (m - m.min()) / (m.max() - m.min() + 1e-12)
        data += [s[gt == 0], s[gt == 1]]
        labels += [f"{method}\nbackground", f"{method}\nanomaly"]
        colors += ["tab:blue", "tab:red"]
    plt.figure(figsize=(1.6 * len(data), 4.5))
    bp = plt.boxplot(data, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    plt.xticks(range(1, len(data) + 1), labels, fontsize=8)
    plt.ylabel("Normalized anomaly score")
    plt.title(f"Background-anomaly separability - {name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=110)
    plt.close()
