import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE
import numpy as np
# Đọc dữ liệu từ file CSV
data = pd.read_csv('creditcard.csv')

# Chuẩn hóa cột 'Amount' và 'Time'
scaler = StandardScaler()
data['Amount_scaled'] = scaler.fit_transform(data[['Amount']])
data['Time_scaled'] = scaler.fit_transform(data[['Time']])

# Xóa các cột 'Class', 'Amount', và 'Time' chưa chuẩn hóa
X = data.drop(columns=['Class', 'Amount', 'Time'])

# Thêm cột 'Amount_scaled' và 'Time_scaled'
X['Amount'] = data['Amount_scaled']
X['Time'] = data['Time_scaled']

# Nhãn dữ liệu
y = data['Class']



# # Sử dụng SMOTE để cân bằng dữ liệu
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X, y)

# # Sau đó chia dữ liệu cân bằng thành tập huấn luyện và kiểm tra
# X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)


# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuyển đổi sang tensor cho PyTorch
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)


import torch.nn as nn
import torch.optim as optim

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2_mu = nn.Linear(32, latent_dim)
        self.fc2_logvar = nn.Linear(32, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 32)
        self.fc4 = nn.Linear(32, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc2_mu(h), self.fc2_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Khởi tạo mô hình VAE
input_dim = X_train.shape[1]
latent_dim = 2
model = VAE(input_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)


def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Huấn luyện mô hình
epochs = 20
model.train()
train_loader = DataLoader(TensorDataset(X_train_tensor, X_train_tensor), batch_size=32, shuffle=True)

for epoch in range(epochs):
    train_loss = 0
    for x, _ in train_loader:
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(x)
        loss = loss_function(recon_batch, x, mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {train_loss / len(train_loader.dataset)}')


# Đặt mô hình vào chế độ đánh giá
model.eval()
with torch.no_grad():
    X_test_pred, _, _ = model(X_test_tensor)
    reconstruction_loss = torch.mean((X_test_tensor - X_test_pred) ** 2, dim=1).numpy()

# Thiết lập ngưỡng phát hiện bất thường
threshold = np.percentile(reconstruction_loss, 95)  # Ngưỡng 95% của lỗi tái tạo

# Gán nhãn dự đoán dựa trên ngưỡng
y_pred = (reconstruction_loss > threshold).astype(int)  # 1 là bất thường, 0 là bình thường

# So sánh kết quả với nhãn thực tế
import pandas as pd

# Tạo DataFrame chứa kết quả dự đoán và nhãn thực tế
results_df = pd.DataFrame({
    'Reconstruction_Loss': reconstruction_loss,   # Lỗi tái tạo từ VAE
    'Predicted_Anomaly': y_pred,                  # 1 là bất thường, 0 là bình thường
    'Actual_Label': y_test.values                 # Nhãn thực tế từ dữ liệu
})

# Hiển thị các giao dịch mà mô hình đánh dấu là gian lận và so sánh với nhãn thực tế
fraud_cases = results_df[results_df['Predicted_Anomaly'] == 1]  # Các giao dịch được dự đoán là gian lận
print(fraud_cases.head())  # Hiển thị một số giao dịch bị dự đoán là gian lận

# Hiển thị các giao dịch mà mô hình bỏ sót (thực tế là gian lận nhưng không được dự đoán)
missed_fraud_cases = results_df[(results_df['Predicted_Anomaly'] == 0) & (results_df['Actual_Label'] == 1)]
print("Missed fraud cases:")
print(missed_fraud_cases.head())


from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# So sánh kết quả với nhãn thực tế
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred))



# Hiển thị các giao dịch được dự đoán là gian lận
fraud_cases = results_df[results_df['Predicted_Anomaly'] == 1]
print("Predicted Fraud Cases:")
print(fraud_cases.head())

# Hiển thị các giao dịch mà mô hình dự đoán sai (False Positives và False Negatives)
false_positives = results_df[(results_df['Predicted_Anomaly'] == 1) & (results_df['Actual_Label'] == 0)]
false_negatives = results_df[(results_df['Predicted_Anomaly'] == 0) & (results_df['Actual_Label'] == 1)]

print("False Positives (Predicted Fraud, but Actually Non-Fraud):")
print(false_positives.head())

print("False Negatives (Predicted Non-Fraud, but Actually Fraud):")
print(false_negatives.head())
