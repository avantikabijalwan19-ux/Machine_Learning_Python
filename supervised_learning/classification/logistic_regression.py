"""
Practical: Logistic Regression for Binary Classification
Description:
    Implements binary classification using the Sigmoid (logistic) function:
    p = 1 / (1 + exp(-(w * x + b)))
    Learns weight (w) and bias (b) via gradient descent to classify samples into Class 0 or 1.
    Accepts interactive user input for training data, labels, and test values.
"""

import math

def parse_float_list(prompt_text, default_list):
    default_str = ", ".join(str(val) for val in default_list)
    raw = input(f"{prompt_text} [default: {default_str}]: ").strip()
    if not raw:
        return default_list
    try:
        parts = raw.replace(",", " ").split()
        return [float(p) for p in parts]
    except ValueError:
        print("  Invalid format. Using default values.")
        return default_list

def sigmoid(z):
    z = max(-500.0, min(500.0, z))
    return 1.0 / (1.0 + math.exp(-z))

def main():
    print("=" * 60)
    print("       LOGISTIC REGRESSION (BINARY CLASSIFICATION)")
    print("=" * 60)

    print("--- Enter Training Dataset ---")
    X = parse_float_list("Enter feature values (e.g. Study Hours)", [1, 2, 3, 6, 7, 8])
    raw_Y = parse_float_list("Enter binary labels (0 or 1)       ", [0, 0, 0, 1, 1, 1])
    Y = [int(y) for y in raw_Y]

    if len(X) != len(Y) or len(X) < 2 or not all(y in (0, 1) for y in Y):
        print("Warning: X and Y must have same length (>=2) and Y must only contain 0 or 1.")
        print("Reverting to default dataset.")
        X = [1.0, 2.0, 3.0, 6.0, 7.0, 8.0]
        Y = [0, 0, 0, 1, 1, 1]

    test_points = parse_float_list("\nEnter test values to classify", [2.5, 4.5, 7.0])

    raw_lr = input("Enter learning rate [default: 0.1]: ").strip()
    learning_rate = float(raw_lr) if raw_lr else 0.1

    raw_epochs = input("Enter epochs [default: 2000]: ").strip()
    epochs = int(raw_epochs) if raw_epochs else 2000

    w = 0.0
    b = 0.0

    print("\n" + "-" * 60)
    print("Training Data:")
    for xi, yi in zip(X, Y):
        print(f"  x = {xi} -> Class {yi}")

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
    print("\nPredictions on Test Values:")
    print(f"{'Input x':<15} {'Probability (Class 1)':<25} {'Predicted Class':<15}")
    print("-" * 60)
    for tx in test_points:
        prob = sigmoid(w * tx + b)
        pred_class = 1 if prob >= 0.5 else 0
        status = f"Class {pred_class}"
        print(f"{tx:<15} {prob:<25.4f} {status:<15}")
    print("=" * 60)

if __name__ == "__main__":
    main()
