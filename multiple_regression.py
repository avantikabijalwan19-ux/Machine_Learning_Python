"""
Practical: Multiple Linear Regression using Gradient Descent
Description:
    Models relationship between multiple independent features X and target y:
    y = b + w1*x1 + w2*x2 + ... + wn*xn
    Optimizes weights and bias using Batch Gradient Descent.
"""

# Training Data: 3 features (e.g. Experience, Test Score, Interview Score)
X = [
    [1, 60, 50],
    [2, 65, 55],
    [3, 70, 60],
    [4, 75, 65],
    [5, 80, 70]
]
y = [50, 55, 60, 65, 70]

n_samples = len(X)
n_features = len(X[0])

# Initialize parameters
weights = [0.0] * n_features
bias = 0.0

learning_rate = 0.0001
epochs = 10000

print("=" * 55)
print("     MULTIPLE LINEAR REGRESSION (GRADIENT DESCENT)")
print("=" * 55)
print(f"Dataset: {n_samples} samples, {n_features} features")

# Gradient Descent Training
for epoch in range(epochs):
    dw = [0.0] * n_features
    db = 0.0

    for i in range(n_samples):
        # Prediction: y_pred = bias + sum(w_j * x_ij)
        pred = bias + sum(weights[j] * X[i][j] for j in range(n_features))
        error = pred - y[i]

        for j in range(n_features):
            dw[j] += error * X[i][j]
        db += error

    # Update weights and bias
    for j in range(n_features):
        weights[j] -= (learning_rate * dw[j]) / n_samples
    bias -= (learning_rate * db) / n_samples

print("\nTrained Model Parameters:")
for j in range(n_features):
    print(f"  Weight w{j + 1} : {weights[j]:.4f}")
print(f"  Bias (b)   : {bias:.4f}")

# Make prediction on new data point
new_x = [6, 85, 75]
pred_new = bias + sum(weights[j] * new_x[j] for j in range(n_features))

print("-" * 55)
print(f"Prediction for input {new_x}: {pred_new:.2f}")
print("=" * 55)

