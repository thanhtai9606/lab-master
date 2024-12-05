from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt

class IsolationForestModel:
    def __init__(self, contamination=0.05, random_state=42):
        self.model = IsolationForest(contamination=contamination, random_state=random_state)

    def train(self, X_train):
        self.model.fit(X_train)

    def predict(self, X_test, threshold_percentile=5):
        anomaly_scores = self.model.decision_function(X_test)
        anomalies = self.model.predict(X_test)
        anomalies = np.where(anomalies == -1, 1, 0)

        threshold = np.percentile(anomaly_scores, threshold_percentile)
        return anomaly_scores, threshold, anomalies

    def visualize(self, anomaly_scores, threshold):
        plt.figure(figsize=(10, 6))
        plt.hist(anomaly_scores, bins=50, alpha=0.75, label='Isolation Forest Anomaly Scores')
        plt.axvline(threshold, color='red', linestyle='dashed', label=f'Threshold: {threshold:.2f}')
        plt.xlabel("Anomaly Score")
        plt.ylabel("Frequency")
        plt.title("Distribution of Isolation Forest Anomaly Scores")
        plt.legend()
        plt.show()
