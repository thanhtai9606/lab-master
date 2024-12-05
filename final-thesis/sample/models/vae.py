import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mean, logvar = torch.chunk(h, 2, dim=-1)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decoder(z)
        return x_hat, mean, logvar

class VAEModel:
    def __init__(self, input_dim, latent_dim=2, lr=0.001):
        self.model = VAE(input_dim, latent_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, X_train, epochs=50, batch_size=32):
        train_loader = torch.utils.data.DataLoader(
            torch.tensor(X_train, dtype=torch.float32), batch_size=batch_size, shuffle=True
        )
        self.model.train()
        for epoch in range(epochs):
            train_loss = 0
            for x in train_loader:
                self.optimizer.zero_grad()
                x_hat, mean, logvar = self.model(x)
                recon_loss = nn.MSELoss()(x_hat, x)
                kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
                loss = recon_loss + kl_loss
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {train_loss / len(train_loader.dataset):.4f}')

    def predict(self, X_test, threshold_percentile=95):
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            X_test_pred, _, _ = self.model(X_test_tensor)
            reconstruction_loss = torch.mean((X_test_tensor - X_test_pred) ** 2, dim=1).numpy()

        threshold = np.percentile(reconstruction_loss, threshold_percentile)
        anomalies = (reconstruction_loss > threshold).astype(int)
        return reconstruction_loss, threshold, anomalies

    def visualize(self, reconstruction_loss, threshold):
        plt.figure(figsize=(10, 6))
        plt.hist(reconstruction_loss, bins=50, alpha=0.75, label='Reconstruction Loss')
        plt.axvline(threshold, color='red', linestyle='dashed', label=f'Threshold: {threshold:.2f}')
        plt.xlabel("Reconstruction Loss")
        plt.ylabel("Frequency")
        plt.title("Distribution of Reconstruction Loss")
        plt.legend()
        plt.show()
