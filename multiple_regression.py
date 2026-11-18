"""
Practical: Multiple Linear Regression using Gradient Descent
Description:
    Models relationship between multiple independent features X and target y:
    y = b + w1*x1 + w2*x2 + ... + wn*xn
    Optimizes weights and bias using Batch Gradient Descent.
    Accepts interactive user input for training samples, prediction inputs, and hyperparameters.
"""

def main():
    print("=" * 60)
    print("     MULTIPLE LINEAR REGRESSION (GRADIENT DESCENT)")
    print("=" * 60)

    # Default dataset
    default_X = [
        [1.0, 60.0, 50.0],
        [2.0, 65.0, 55.0],
        [3.0, 70.0, 60.0],
        [4.0, 75.0, 65.0],
        [5.0, 80.0, 70.0]
    ]
    default_y = [50.0, 55.0, 60.0, 65.0, 70.0]
    default_new_x = [6.0, 85.0, 75.0]

    choice = input("Do you want to enter a custom dataset? (y/n) [default: n]: ").strip().lower()

    if choice == 'y':
        try:
            raw_f = input("Enter number of features [default: 3]: ").strip()
            n_features = int(raw_f) if raw_f else 3

            raw_s = input("Enter number of training samples [default: 5]: ").strip()
            n_samples = int(raw_s) if raw_s else 5

            X = []
            y = []
            print(f"Enter each sample as {n_features} feature values followed by 1 target value:")
            print("Example format: 1.0, 60.0, 50.0, 50.0")

            for i in range(n_samples):
                row_raw = input(f"  Sample {i + 1}: ").strip().replace(",", " ")
                parts = [float(p) for p in row_raw.split()]
                if len(parts) != n_features + 1:
                    raise ValueError(f"Expected {n_features + 1} values, got {len(parts)}")
                X.append(parts[:n_features])
                y.append(parts[-1])
        except Exception as e:
            print(f"Invalid input ({e}). Reverting to default dataset.")
            X = default_X
            y = default_y
            n_features = len(X[0])
            n_samples = len(X)
    else:
        X = default_X
        y = default_y
        n_features = len(X[0])
        n_samples = len(X)

    # Hyperparameters
    raw_lr = input("Enter learning rate [default: 0.0001]: ").strip()
    learning_rate = float(raw_lr) if raw_lr else 0.0001

    raw_epochs = input("Enter number of epochs [default: 10000]: ").strip()
    epochs = int(raw_epochs) if raw_epochs else 10000

    # Prediction point
    default_new_x_str = ", ".join(str(v) for v in default_new_x[:n_features])
    raw_new_x = input(f"Enter feature values to predict ({n_features} values) [default: {default_new_x_str}]: ").strip()
    if raw_new_x:
        try:
            new_x = [float(p) for p in raw_new_x.replace(",", " ").split()]
            if len(new_x) != n_features:
                print(f"Warning: Expected {n_features} values. Using default prediction input.")
                new_x = default_new_x[:n_features] if len(default_new_x) >= n_features else [1.0] * n_features
        except ValueError:
            print("Invalid input. Using default prediction values.")
            new_x = default_new_x[:n_features] if len(default_new_x) >= n_features else [1.0] * n_features
    else:
        new_x = default_new_x[:n_features] if len(default_new_x) >= n_features else [1.0] * n_features

    print("\n" + "-" * 60)
    print(f"Training Dataset ({n_samples} samples, {n_features} features):")
    for i in range(n_samples):
        print(f"  Sample {i + 1}: X = {X[i]}  -->  y = {y[i]}")

    # Initialize parameters
    weights = [0.0] * n_features
    bias = 0.0

    # Gradient Descent Training
    for epoch in range(epochs):
        dw = [0.0] * n_features
        db = 0.0

        for i in range(n_samples):
            pred = bias + sum(weights[j] * X[i][j] for j in range(n_features))
            error = pred - y[i]

            for j in range(n_features):
                dw[j] += error * X[i][j]
            db += error

        for j in range(n_features):
            weights[j] -= (learning_rate * dw[j]) / n_samples
        bias -= (learning_rate * db) / n_samples

    print("\nTrained Model Parameters:")
    for j in range(n_features):
        print(f"  Weight w{j + 1} : {weights[j]:.4f}")
    print(f"  Bias (b)   : {bias:.4f}")

    # Make prediction on new data point
    pred_new = bias + sum(weights[j] * new_x[j] for j in range(n_features))

    print("-" * 60)
    print(f"Prediction for input {new_x}: {pred_new:.2f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
