"""Spectral-spatial convolutional autoencoder.

The whole image is treated as a multi-channel input (bands = channels).
Convolutions let the model use each pixel's spatial neighbourhood, so the
reconstruction reflects local background structure. Pixels that do not fit
their surroundings are reconstructed poorly and get a high anomaly score.
"""
import numpy as np
import torch
import torch.nn as nn


class ConvAE(nn.Module):
    def __init__(self, n_bands, hidden=64, latent=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(n_bands, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.LeakyReLU(0.2),
            nn.Conv2d(hidden, latent, 3, padding=1), nn.BatchNorm2d(latent), nn.LeakyReLU(0.2),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.LeakyReLU(0.2),
            nn.Conv2d(hidden, n_bands, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def conv_ae_detector(cube, iters=500, lr=1e-3, seed=0, verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H, W, B = cube.shape
    x = torch.tensor(cube.transpose(2, 0, 1)[None], dtype=torch.float32, device=device)

    model = ConvAE(B).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for it in range(iters):
        loss = ((model(x) - x) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if verbose and (it + 1) % 100 == 0:
            print(f"    iter {it + 1}/{iters}  loss={loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        err = ((model(x) - x) ** 2).mean(dim=1)[0]
    return err.cpu().numpy()
