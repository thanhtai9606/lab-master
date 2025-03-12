import matplotlib.pyplot as plt
import numpy as np

def plot_results(test_probs, threshold):
    # Vẽ phân phối xác suất
    plt.figure(figsize=(10, 6))
    plt.hist(test_probs, bins=50, alpha=0.75, label='Discriminator Probabilities')
    plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold ({threshold:.2f})')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.title('Distribution of Discriminator Probabilities on Test Data')
    plt.legend()
    plt.show()

def plot_anomalies(test_probs, anomalies, threshold):
    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(test_probs)), test_probs, label='Probabilities')
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.2f}')
    plt.scatter(np.where(anomalies == 1), test_probs[anomalies == 1], color='orange', label='Anomalies')
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Probability')
    plt.title('Anomalies Detected in Test Data')
    plt.show()