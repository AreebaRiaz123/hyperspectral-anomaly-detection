import numpy as np


# Global RX: distance of each pixel from mean of whole image
def rx_detector(cube):
    H, W, B = cube.shape
    X = cube.reshape(-1, B)
    Xc = X - X.mean(axis=0)
    cov_inv = np.linalg.pinv(np.cov(Xc, rowvar=False))
    scores = np.sum((Xc @ cov_inv) * Xc, axis=1)  # mahalanobis distance
    return scores.reshape(H, W)


# reduce bands with PCA (needed for local RX, otherwise covariance is unstable)
def pca_reduce(cube, n_components=20):
    H, W, B = cube.shape
    X = cube.reshape(-1, B)
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return (Xc @ Vt[:n_components].T).reshape(H, W, n_components)


# Local RX with dual window
# background = pixels inside outer window but outside inner window
def local_rx_detector(cube, inner=3, outer=11, n_components=20, eps=1e-6):
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
            win = padded[i:i + outer, j:j + outer][mask]
            mu = win.mean(axis=0)
            cov = np.cov(win, rowvar=False) + eps * np.eye(D)
            d = X[i, j] - mu
            scores[i, j] = d @ np.linalg.solve(cov, d)
    return scores
