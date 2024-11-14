import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import os

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
    def __init__(self, input_dim, latent_dim=2, model_path="vae_model.pth"):
        self.model = VAE(input_dim, latent_dim)
        self.model_path = model_path
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
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

        # Lưu mô hình sau khi huấn luyện
        torch.save(self.model.state_dict(), self.model_path)
        print("Mô hình VAE đã được lưu thành công.")

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.eval()  # Chuyển sang chế độ dự đoán
            print("Mô hình VAE đã được tải thành công.")
        else:
            raise FileNotFoundError("Mô hình chưa được huấn luyện. Hãy huấn luyện mô hình trước.")

    def predict(self, features, true_labels=None):
        # Tải mô hình nếu chưa được tải
        self.load_model()
        
        X_tensor = torch.tensor(features, dtype=torch.float32)
        with torch.no_grad():
            x_hat, _, _ = self.model(X_tensor)
            recon_loss = torch.mean((X_tensor - x_hat) ** 2, dim=1).numpy()

        threshold = np.percentile(recon_loss, 95)
        predictions = (recon_loss > threshold).astype(int)

        # Nếu `true_labels` được cung cấp, tính toán các chỉ số đánh giá
        if true_labels is not None:
            cm = confusion_matrix(true_labels, predictions)
            report = classification_report(true_labels, predictions, output_dict=True)
            roc_auc = roc_auc_score(true_labels, predictions)

            return {
                "predictions": list(predictions),  # Đảm bảo predictions là danh sách
                "anomaly_scores": list(recon_loss),  # Đảm bảo anomaly_scores là danh sách
                "threshold": threshold,
                "confusion_matrix": cm.tolist(),  # Chuyển cm sang danh sách
                "classification_report": report,
                "roc_auc_score": roc_auc
            }
        else:
            # Nếu không có `true_labels`, chỉ trả về dự đoán và điểm bất thường
            return {
                "predictions": list(predictions),  # Đảm bảo predictions là danh sách
                "anomaly_scores": list(recon_loss),  # Đảm bảo anomaly_scores là danh sách
                "threshold": threshold
            }