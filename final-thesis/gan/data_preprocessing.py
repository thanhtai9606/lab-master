import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

def preprocess_data(file_path):
    # Đọc dữ liệu
    data = pd.read_csv(file_path)
    
    # Kiểm tra và xử lý dữ liệu bị thiếu
    data.fillna(data.mean(), inplace=True)
    
    # Bỏ cột không cần thiết
    features = data.drop(columns=['Time'], errors='ignore')  # Bỏ cột 'Time' nếu có
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Chia dữ liệu: 80% huấn luyện, 20% kiểm tra
    train_size = int(0.8 * len(scaled_features))
    X_train = scaled_features[:train_size]
    X_test = scaled_features[train_size:]
    
    # Chuyển dữ liệu thành tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    latent_dim = 10  # Chiều không gian ẩn
    input_dim = X_train.shape[1]  # Số lượng đặc trưng trong dữ liệu
    
    return X_train_tensor, X_test_tensor, latent_dim, input_dim