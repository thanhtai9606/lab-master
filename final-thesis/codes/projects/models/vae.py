# models/vae.py
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

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
    def __init__(self, input_dim, latent_dim=2):
        self.model = VAE(input_dim, latent_dim)
        self.latent_dim = latent_dim

    def train(self, features, epochs=20):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        X_tensor = torch.tensor(features, dtype=torch.float32)

        for epoch in range(epochs):
            optimizer.zero_grad()
            x_hat, mean, logvar = self.model(X_tensor)
            recon_loss = nn.MSELoss()(x_hat, X_tensor)
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss
            loss.backward()
            optimizer.step()

    def predict(self, features, true_labels):
        X_tensor = torch.tensor(features, dtype=torch.float32)
        with torch.no_grad():
            x_hat, _, _ = self.model(X_tensor)
            recon_loss = torch.mean((X_tensor - x_hat) ** 2, dim=1).numpy()
        
        threshold = np.percentile(recon_loss, 95)
        predictions = (recon_loss > threshold).astype(int)
        
        cm = confusion_matrix(true_labels, predictions)
        report = classification_report(true_labels, predictions, output_dict=True)
        roc_auc = roc_auc_score(true_labels, predictions)

        return {
            "predictions": predictions,
            "anomaly_scores": recon_loss,
            "threshold": threshold,
            "confusion_matrix": cm,
            "classification_report": report,
            "roc_auc_score": roc_auc
        }