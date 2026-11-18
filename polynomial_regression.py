"""
Practical: Polynomial Regression from Scratch
Description:
    Models non-linear relationships by fitting a polynomial curve of degree d:
    y = w0 + w1*x + w2*x^2 + ... + wd*x^d
    Uses gradient descent to optimize polynomial coefficients.
"""

# Non-linear quadratic training data: y = x^2
X = [1, 2, 3, 4, 5]
Y = [1, 4, 9, 16, 25]

degree = 2
weights = [0.0] * (degree + 1)
learning_rate = 0.001
epochs = 15000

print("=" * 55)
print("       POLYNOMIAL REGRESSION (DEGREE = 2)")
print("=" * 55)
print("Training Data: (x, y)")
for xi, yi in zip(X, Y):
    print(f"  ({xi}, {yi})")

# Gradient Descent
for epoch in range(epochs):
    gradients = [0.0] * (degree + 1)

    for i in range(len(X)):
        x = X[i]
        actual = Y[i]

        # Calculate prediction: sum(w_j * x^j)
        pred = sum(weights[j] * (x ** j) for j in range(degree + 1))
        error = pred - actual

        for j in range(degree + 1):
            gradients[j] += error * (x ** j)

    # Update polynomial coefficients
    for j in range(degree + 1):
        weights[j] -= (learning_rate * gradients[j]) / len(X)

print("\nLearned Polynomial Coefficients:")
for j in range(degree + 1):
    print(f"  w{j} (x^{j} coeff) : {weights[j]:.4f}")

poly_terms = " + ".join([f"{weights[j]:.2f}*x^{j}" if j > 1 else (f"{weights[j]:.2f}*x" if j == 1 else f"{weights[j]:.2f}") for j in range(degree + 1)])
print(f"Fitted Curve: y = {poly_terms}")

# Predict for new value
new_x = 6
pred_new = sum(weights[j] * (new_x ** j) for j in range(degree + 1))
print("-" * 55)
print(f"Prediction for x = {new_x}: {pred_new:.2f} (Actual: {new_x**2})")
print("=" * 55)

