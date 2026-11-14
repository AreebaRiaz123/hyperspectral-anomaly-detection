import os
import urllib.request

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score, roc_curve

DATA_URL = "https://raw.githubusercontent.com/jeline0110/GT-HAD/main/data/{}.mat"


# download .mat file if not already there
def download_dataset(name, folder="data"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.mat")
    if not os.path.exists(path):
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(DATA_URL.format(name), path)
    return path


# returns data cube (H x W x B) and ground truth map
def load_hsi(path):
    mat = loadmat(path)
    data = mat["data"].astype(np.float64)
    gt = (mat["map"] > 0).astype(int)
    return data, gt


def normalize(data):
    return (data - data.min()) / (data.max() - data.min())


def evaluate(score_map, gt):
    y_true, y_score = gt.ravel(), score_map.ravel()
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc, fpr, tpr


# pick 3 bands to show image as RGB
def false_color(data):
    B = data.shape[2]
    rgb = data[:, :, [int(B * 0.75), int(B * 0.5), int(B * 0.25)]]
    return (rgb - rgb.min()) / (rgb.max() - rgb.min())


def plot_detection_maps(name, data, gt, maps, aucs, save_path):
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


# box plot: background vs anomaly scores (bigger gap = better)
def plot_separability(name, maps, gt, save_path):
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
