import numpy as np
import torch
import torch.nn as nn

# Autoencoder learns to reconstruct background spectra.
# Anomalies are rare so their reconstruction error is high.


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class SpectralAE(nn.Module):
    def __init__(self, n_bands, hidden=64, latent=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_bands, hidden), nn.ReLU(),
            nn.Linear(hidden, latent), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, n_bands), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_ae(X_train, n_bands, epochs=100, lr=1e-3, batch_size=256, verbose=False):
    model = SpectralAE(n_bands).to(X_train.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        perm = torch.randperm(X_train.shape[0], device=X_train.device)
        total = 0.0
        for i in range(0, X_train.shape[0], batch_size):
            batch = X_train[perm[i:i + batch_size]]
            loss = loss_fn(model(batch), batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * batch.shape[0]
        if verbose and (epoch + 1) % 25 == 0:
            print(f"    epoch {epoch + 1}/{epochs}  loss={total / X_train.shape[0]:.6f}")
    return model.eval()


def reconstruction_error(model, X):
    with torch.no_grad():
        return ((model(X) - X) ** 2).mean(dim=1)


# simple AE, trained on all pixels
def ae_detector(cube, epochs=100, seed=0, verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    H, W, B = cube.shape
    X = torch.tensor(cube.reshape(-1, B), dtype=torch.float32, device=get_device())
    model = train_ae(X, B, epochs=epochs, verbose=verbose)
    return reconstruction_error(model, X).cpu().numpy().reshape(H, W)


# AE with background purification
# step 1: train on all pixels
# step 2: remove top 5% error pixels (probably anomalies) and train again
def purified_ae_detector(cube, epochs=100, remove_ratio=0.05, seed=0, verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    H, W, B = cube.shape
    X = torch.tensor(cube.reshape(-1, B), dtype=torch.float32, device=get_device())

    model1 = train_ae(X, B, epochs=epochs, verbose=verbose)
    err1 = reconstruction_error(model1, X)
    keep = err1 <= torch.quantile(err1, 1 - remove_ratio)

    model2 = train_ae(X[keep], B, epochs=epochs, verbose=verbose)
    return reconstruction_error(model2, X).cpu().numpy().reshape(H, W)
