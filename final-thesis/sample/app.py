from models.gan import GAN
from models.isolation_forest import IsolationForestModel
from models.vae import VAEModel
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Đọc dữ liệu
data = pd.read_csv('../../data/Chiller.csv')
features = data.drop(columns=['Time'], errors='ignore')
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Chọn thuật toán
algorithm = "gan"
if algorithm == "gan":
    model = GAN(input_dim=scaled_features.shape[1])
    model.train(scaled_features)
    probs, threshold, anomalies = model.predict(scaled_features)
    model.visualize(probs, threshold)
elif algorithm == "isolation_forest":
    model = IsolationForestModel()
    model.train(scaled_features)
    scores, threshold, anomalies = model.predict(scaled_features)
    model.visualize(scores, threshold)
elif algorithm == "vae":
    model = VAEModel(input_dim=scaled_features.shape[1])
    model.train(scaled_features)
    losses, threshold, anomalies = model.predict(scaled_features)
    model.visualize(losses, threshold)
