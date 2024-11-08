# models/isolation_forest.py
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

class IsolationForestModel:
    def __init__(self):
        self.model = IsolationForest(contamination=0.1, random_state=42)

    def train(self, features):
        self.model.fit(features)

    def predict(self, features, true_labels):
        predictions = self.model.predict(features)
        anomaly_scores = self.model.decision_function(features)
        predictions = (predictions == -1).astype(int)
        
        cm = confusion_matrix(true_labels, predictions)
        report = classification_report(true_labels, predictions, output_dict=True)
        roc_auc = roc_auc_score(true_labels, predictions)
        
        return {
            "predictions": predictions,
            "anomaly_scores": anomaly_scores,
            "confusion_matrix": cm,
            "classification_report": report,
            "roc_auc_score": roc_auc
        }