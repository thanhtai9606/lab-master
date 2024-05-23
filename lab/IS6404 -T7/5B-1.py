import numpy as np
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt

# Load the CSV data
file_path = './data/gdp.csv'
data = pd.read_csv(file_path)

# Extract features and target
X = data['Year'].values.reshape(1, -1)
Y = data['GDP'].values.reshape(1, -1)

# Normalize the data
X = (X - np.mean(X)) / np.std(X)
Y = (Y - np.mean(Y)) / np.std(Y)

# Initialize parameters
def initialize_parameters(input_size, hidden_size, output_size):
    W1 = np.random.randn(hidden_size, input_size) * 0.01
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros((output_size, 1))
    return W1, b1, W2, b2

# Sigmoid activation function
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

# Derivative of sigmoid function
def sigmoid_derivative(Z):
    return sigmoid(Z) * (1 - sigmoid(Z))

# Forward propagation
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

# Compute cost
def compute_cost(A2, Y):
    m = Y.shape[1]
    cost = -1/m * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
    return cost

# Backward propagation
def backward_propagation(X, Y, Z1, A1, Z2, A2, W2):
    m = X.shape[1]
    dZ2 = A2 - Y
    dW2 = 1/m * np.dot(dZ2, A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * sigmoid_derivative(Z1)
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

# Update parameters
def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    return W1, b1, W2, b2

# Model training
def model(X, Y, input_size, hidden_size, output_size, learning_rate, epochs):
    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)
    costs = []

    for epoch in range(epochs):
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
        cost = compute_cost(A2, Y)
        dW1, db1, dW2, db2 = backward_propagation(X, Y, Z1, A1, Z2, A2, W2)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

        if epoch % 100 == 0:
            costs.append(cost)
            print(f"Cost after epoch {epoch}: {cost}")

    return W1, b1, W2, b2, costs

# Export results to Excel
def export_to_excel(costs, filename='results.xlsx'):
    df = pd.DataFrame(costs, columns=['Cost'])
    df.to_excel(filename, index=False)

# Train the model
W1, b1, W2, b2, costs = model(X, Y, input_size=X.shape[0], hidden_size=4, output_size=1, learning_rate=0.01, epochs=1000)

# Export the costs to Excel
export_to_excel(costs, filename='./data/results.xlsx')

# Plot the cost over epochs
plt.plot(costs)
plt.xlabel('Epoch (x100)')
plt.ylabel('Cost')
plt.title('Training Cost Over Time')
plt.show()
