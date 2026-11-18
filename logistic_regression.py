"""
Practical: Logistic Regression for Binary Classification
Description:
    Implements binary classification using the Sigmoid (logistic) function:
    p = 1 / (1 + exp(-(w * x + b)))
    Learns weight (w) and bias (b) via gradient descent to classify samples into Class 0 or 1.
"""

import math

# Training data: Study hours -> Pass (1) or Fail (0)
X = [1, 2, 3, 6, 7, 8]
Y = [0, 0, 0, 1, 1, 1]

w = 0.0
b = 0.0
learning_rate = 0.1
epochs = 2000

def sigmoid(z):
    # Clamp z to avoid overflow
    z = max(-500.0, min(500.0, z))
    return 1.0 / (1.0 + math.exp(-z))

print("=" * 55)
print("       LOGISTIC REGRESSION (BINARY CLASSIFICATION)")
print("=" * 55)
print("Training Data (Hours -> Pass/Fail):")
for xi, yi in zip(X, Y):
    print(f"  {xi} hours -> {'Pass (1)' if yi == 1 else 'Fail (0)'}")

# Gradient Descent
for epoch in range(epochs):
    dw = 0.0
    db = 0.0

    for i in range(len(X)):
        z = w * X[i] + b
        pred = sigmoid(z)
        error = pred - Y[i]

        dw += error * X[i]
        db += error

    w -= learning_rate * (dw / len(X))
    b -= learning_rate * (db / len(X))

print(f"\nTrained Parameters: Weight = {w:.4f}, Bias = {b:.4f}")

# Predictions on test points
test_points = [2.5, 4.5, 7.0]
print("\nPredictions on Test Values:")
print(f"{'Input Hours':<15} {'Probability (Pass)':<25} {'Predicted Class':<15}")
print("-" * 55)
for tx in test_points:
    prob = sigmoid(w * tx + b)
    pred_class = 1 if prob >= 0.5 else 0
    status = "Pass (1)" if pred_class == 1 else "Fail (0)"
    print(f"{tx:<15} {prob:<25.4f} {status:<15}")
print("=" * 55)

