import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class GAN:
    def __init__(self, input_dim, latent_dim=10, lr=0.001):
        self.generator = Generator(latent_dim, input_dim)
        self.discriminator = Discriminator(input_dim)
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=lr)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr)
        self.latent_dim = latent_dim

    def train(self, X_train, epochs=50, batch_size=32):
        train_loader = torch.utils.data.DataLoader(
            torch.tensor(X_train, dtype=torch.float32), batch_size=batch_size, shuffle=True
        )
        for epoch in range(epochs):
            for real_data in train_loader:
                # Huấn luyện Discriminator
                self.optimizer_d.zero_grad()
                real_labels = torch.ones((real_data.size(0), 1))
                fake_labels = torch.zeros((real_data.size(0), 1))

                noise = torch.randn(real_data.size(0), self.latent_dim)
                fake_data = self.generator(noise)

                real_loss = nn.BCELoss()(self.discriminator(real_data), real_labels)
                fake_loss = nn.BCELoss()(self.discriminator(fake_data.detach()), fake_labels)

                d_loss = real_loss + fake_loss
                d_loss.backward()
                self.optimizer_d.step()

                # Huấn luyện Generator
                self.optimizer_g.zero_grad()
                generated_data = self.generator(noise)
                g_loss = nn.BCELoss()(self.discriminator(generated_data), real_labels)
                g_loss.backward()
                self.optimizer_g.step()

            print(f'Epoch [{epoch + 1}/{epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')

    def predict(self, X_test, threshold_percentile=5):
        self.discriminator.eval()
        with torch.no_grad():
            test_probs = self.discriminator(torch.tensor(X_test, dtype=torch.float32)).numpy().flatten()

        threshold = np.percentile(test_probs, threshold_percentile)
        anomalies = (test_probs < threshold).astype(int)
        return test_probs, threshold, anomalies

    def visualize(self, test_probs, threshold):
        plt.figure(figsize=(10, 6))
        plt.hist(test_probs, bins=50, alpha=0.75, label='GAN Probabilities')
        plt.axvline(threshold, color='red', linestyle='dashed', label=f'Threshold: {threshold:.2f}')
        plt.xlabel("Probability")
        plt.ylabel("Frequency")
        plt.title("Distribution of GAN Probabilities on Test Data")
        plt.legend()
        plt.show()
