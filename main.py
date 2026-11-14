"""Run all anomaly detectors on benchmark scenes and save results.

Usage:
    python main.py                          # all methods, all datasets
    python main.py --datasets gulfport      # one dataset
    python main.py --methods RX AE          # selected methods
"""
import argparse
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from utils import (download_dataset, load_hsi, normalize, evaluate,
                   plot_detection_maps, plot_roc, plot_separability)
from rx_detector import rx_detector, local_rx_detector
from autoencoder import ae_detector, purified_ae_detector
from conv_autoencoder import conv_ae_detector

ALL_DATASETS = ["gulfport", "los-angeles-1", "los-angeles-2",
                "texas-goast", "cat-island", "pavia"]

METHODS = {
    "RX": rx_detector,
    "LRX": local_rx_detector,
    "AE": ae_detector,
    "AE-purified": purified_ae_detector,
    "ConvAE": conv_ae_detector,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    table = {}
    for name in args.datasets:
        data, gt = load_hsi(download_dataset(name))
        data = normalize(data)
        H, W, B = data.shape
        print(f"\n=== {name}: {H}x{W}x{B}, anomaly pixels = {gt.sum()} ===")

        maps, aucs, rocs = {}, {}, {}
        for method in args.methods:
            t0 = time.time()
            maps[method] = METHODS[method](data)
            auc, fpr, tpr = evaluate(maps[method], gt)
            aucs[method], rocs[method] = auc, (fpr, tpr)
            print(f"  {method:12s} AUC = {auc:.4f}   ({time.time() - t0:.1f}s)")

        plot_detection_maps(name, data, gt, maps, aucs, f"{args.out}/{name}_maps.png")
        plot_roc(name, rocs, aucs, f"{args.out}/{name}_roc.png")
        plot_separability(name, maps, gt, f"{args.out}/{name}_boxplot.png")
        table[name] = aucs

    # AUC table (markdown), best value per row in bold
    lines = ["| Dataset | " + " | ".join(args.methods) + " |",
             "|---" * (len(args.methods) + 1) + "|"]
    for name, aucs in table.items():
        best = max(aucs.values())
        cells = [f"**{aucs[m]:.4f}**" if aucs[m] == best else f"{aucs[m]:.4f}" for m in args.methods]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    means = [sum(t[m] for t in table.values()) / len(table) for m in args.methods]
    lines.append("| **Average** | " + " | ".join(f"{v:.4f}" for v in means) + " |")
    with open(f"{args.out}/auc_table.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
