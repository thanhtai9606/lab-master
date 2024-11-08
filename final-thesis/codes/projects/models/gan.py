# models/gan.py
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

class GAN_Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(GAN_Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class GAN_Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(GAN_Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class GANModel:
    def __init__(self, input_dim, latent_dim=10):
        self.generator = GAN_Generator(latent_dim, input_dim)
        self.discriminator = GAN_Discriminator(input_dim)
        self.latent_dim = latent_dim

    def train(self, features, epochs=20, batch_size=32):
        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=0.001)
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=0.001)
        X_tensor = torch.tensor(features, dtype=torch.float32)
        train_loader = torch.utils.data.DataLoader(X_tensor, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for real_data in train_loader:
                optimizer_d.zero_grad()
                real_labels = torch.ones(real_data.size(0), 1)
                fake_labels = torch.zeros(real_data.size(0), 1)
                noise = torch.randn(real_data.size(0), self.latent_dim)
                fake_data = self.generator(noise)

                real_loss = nn.BCELoss()(self.discriminator(real_data), real_labels)
                fake_loss = nn.BCELoss()(self.discriminator(fake_data.detach()), fake_labels)
                d_loss = real_loss + fake_loss
                d_loss.backward()
                optimizer_d.step()

                optimizer_g.zero_grad()
                generated_data = self.generator(noise)
                g_loss = nn.BCELoss()(self.discriminator(generated_data), real_labels)
                g_loss.backward()
                optimizer_g.step()

    def predict(self, features, true_labels):
        X_tensor = torch.tensor(features, dtype=torch.float32)
        with torch.no_grad():
            anomaly_scores = self.discriminator(X_tensor).numpy().flatten()
        
        threshold = 0.5
        predictions = (anomaly_scores < threshold).astype(int)
        
        cm = confusion_matrix(true_labels, predictions)
        report = classification_report(true_labels, predictions, output_dict=True)
        roc_auc = roc_auc_score(true_labels, predictions)

        return {
            "predictions": predictions,
            "anomaly_scores": anomaly_scores,
            "threshold": threshold,
            "confusion_matrix": cm,
            "classification_report": report,
            "roc_auc_score": roc_auc
        }