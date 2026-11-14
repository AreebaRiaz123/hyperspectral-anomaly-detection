"""Classical RX-family anomaly detectors.

- Global RX: background = whole image.
- Local RX (dual window): background = ring of pixels around each pixel,
  excluding an inner guard window that may contain the target itself.
"""
import numpy as np


def rx_detector(cube):
    """Global Reed-Xiaoli detector: Mahalanobis distance to the image mean."""
    H, W, B = cube.shape
    X = cube.reshape(-1, B)
    Xc = X - X.mean(axis=0)
    cov_inv = np.linalg.pinv(np.cov(Xc, rowvar=False))
    scores = np.sum((Xc @ cov_inv) * Xc, axis=1)
    return scores.reshape(H, W)


def pca_reduce(cube, n_components=20):
    """Project spectra onto their first principal components."""
    H, W, B = cube.shape
    X = cube.reshape(-1, B)
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return (Xc @ Vt[:n_components].T).reshape(H, W, n_components)


def local_rx_detector(cube, inner=3, outer=11, n_components=20, eps=1e-6):
    """Dual-window local RX.

    For every pixel, the background statistics are estimated from the
    pixels inside the outer window but outside the inner (guard) window.
    PCA is applied first to keep the local covariance well conditioned.
    """
    X = pca_reduce(cube, n_components)
    H, W, D = X.shape
    r_in, r_out = inner // 2, outer // 2
    padded = np.pad(X, ((r_out, r_out), (r_out, r_out), (0, 0)), mode="reflect")

    mask = np.ones((outer, outer), dtype=bool)
    c = r_out
    mask[c - r_in:c + r_in + 1, c - r_in:c + r_in + 1] = False

    scores = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            win = padded[i:i + outer, j:j + outer][mask]       # background ring
            mu = win.mean(axis=0)
            cov = np.cov(win, rowvar=False) + eps * np.eye(D)
            d = X[i, j] - mu
            scores[i, j] = d @ np.linalg.solve(cov, d)
    return scores
