import numpy as np
import pandas as pd
import openpyxl

# Load the CSV data
file_path = './data/gdp.csv'
data = pd.read_csv(file_path)

# Normalize data
data['Year'] = (data['Year'] - data['Year'].mean()) / data['Year'].std()
data['GDP'] = (data['GDP'] - data['GDP'].mean()) / data['GDP'].std()

# Extract features and target
X = data['Year'].values.reshape(-1, 1)
Y = data['GDP'].values.reshape(-1, 1)

# Initialize weights and biases
W1 = np.random.randn(1, 1)
b1 = np.zeros((1, 1))
W2 = np.random.randn(1, 1)
b2 = np.zeros((1, 1))

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Learning rate
learning_rate = 0.01

# Training the neural network
costs = []
epochs = 10000
for epoch in range(epochs):
    # Forward pass
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    
    # Calculate the cost (Mean Squared Error)
    cost = np.mean((A2 - Y) ** 2)
    costs.append(cost)
    
    # Backward pass
    dZ2 = A2 - Y
    dW2 = np.dot(A1.T, dZ2) / len(X)
    db2 = np.sum(dZ2) / len(X)
    
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / len(X)
    db1 = np.sum(dZ1) / len(X)
    
    # Update weights and biases
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Cost: {cost}")

# Export results to Excel
df = pd.DataFrame(costs, columns=['Cost'])
df.to_excel('./data/results.xlsx', index=False)

# Plot the cost over epochs
import matplotlib.pyplot as plt

plt.plot(costs)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Training Cost Over Time')
plt.show()
