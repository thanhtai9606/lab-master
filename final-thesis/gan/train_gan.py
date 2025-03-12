import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

def train_gan(generator, discriminator, X_train_tensor, latent_dim, epochs=50, batch_size=32):
    optimizer_g = optim.Adam(generator.parameters(), lr=0.001)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.001)
    train_loader = torch.utils.data.DataLoader(X_train_tensor, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        for real_data in train_loader:
            # Huấn luyện Discriminator
            optimizer_d.zero_grad()
            real_labels = torch.ones((real_data.size(0), 1))  # Nhãn thật
            fake_labels = torch.zeros((real_data.size(0), 1))  # Nhãn giả

            noise = torch.randn(real_data.size(0), latent_dim)  # Sinh noise ngẫu nhiên
            fake_data = generator(noise)  # Tạo dữ liệu giả

            real_loss = nn.BCELoss()(discriminator(real_data), real_labels)
            fake_loss = nn.BCELoss()(discriminator(fake_data.detach()), fake_labels)

            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_d.step()

            # Huấn luyện Generator
            optimizer_g.zero_grad()
            generated_data = generator(noise)
            g_loss = nn.BCELoss()(discriminator(generated_data), real_labels)
            g_loss.backward()
            optimizer_g.step()

        print(f'Epoch [{epoch + 1}/{epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')

def detect_anomalies(discriminator, X_test_tensor, threshold_percentile=5):
    discriminator.eval()
    with torch.no_grad():
        test_probs = discriminator(X_test_tensor).numpy().flatten()

    # Tính ngưỡng
    threshold = np.percentile(test_probs, threshold_percentile)
    anomalies = (test_probs < threshold).astype(int)  # Đánh dấu bất thường nếu xác suất < ngưỡng
    
    return test_probs, anomalies, threshold