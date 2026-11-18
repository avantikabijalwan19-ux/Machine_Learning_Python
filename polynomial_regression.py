"""
Practical: Polynomial Regression from Scratch
Description:
    Models non-linear relationships by fitting a polynomial curve of degree d:
    y = w0 + w1*x + w2*x^2 + ... + wd*x^d
    Uses gradient descent to optimize polynomial coefficients.
    Accepts interactive user input for polynomial degree, training data, and query points.
"""

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

def main():
    print("=" * 60)
    print("       POLYNOMIAL REGRESSION (FROM SCRATCH)")
    print("=" * 60)

    # Degree input
    raw_deg = input("Enter polynomial degree [default: 2]: ").strip()
    try:
        degree = int(raw_deg) if raw_deg else 2
        if degree < 1:
            print("Degree must be at least 1. Setting degree to 2.")
            degree = 2
    except ValueError:
        print("Invalid degree. Using default degree = 2.")
        degree = 2

    # Data inputs
    print("\n--- Enter Training Dataset ---")
    X = parse_float_list("Enter X values", [1, 2, 3, 4, 5])
    Y = parse_float_list("Enter Y values", [1, 4, 9, 16, 25])

    if len(X) != len(Y) or len(X) <= degree:
        print(f"Warning: Number of data points ({len(X)}) must equal Y length and be > degree ({degree}).")
        print("Reverting to default quadratic data: X = [1, 2, 3, 4, 5], Y = [1, 4, 9, 16, 25]")
        X = [1.0, 2.0, 3.0, 4.0, 5.0]
        Y = [1.0, 4.0, 9.0, 16.0, 25.0]

    # Query input
    raw_new_x = input("\nEnter new x value to predict [default: 6]: ").strip()
    try:
        new_x = float(raw_new_x) if raw_new_x else 6.0
    except ValueError:
        print("Invalid input. Using default x = 6.0")
        new_x = 6.0

    raw_lr = input("Enter learning rate [default: 0.001]: ").strip()
    learning_rate = float(raw_lr) if raw_lr else 0.001

    raw_epochs = input("Enter training epochs [default: 15000]: ").strip()
    epochs = int(raw_epochs) if raw_epochs else 15000

    weights = [0.0] * (degree + 1)

    print("\n" + "-" * 60)
    print("Training Data: (x, y)")
    for xi, yi in zip(X, Y):
        print(f"  ({xi}, {yi})")

    # Gradient Descent
    for epoch in range(epochs):
        gradients = [0.0] * (degree + 1)

        for i in range(len(X)):
            x = X[i]
            actual = Y[i]

            pred = sum(weights[j] * (x ** j) for j in range(degree + 1))
            error = pred - actual

            for j in range(degree + 1):
                gradients[j] += error * (x ** j)

        for j in range(degree + 1):
            weights[j] -= (learning_rate * gradients[j]) / len(X)

    print("\nLearned Polynomial Coefficients:")
    for j in range(degree + 1):
        print(f"  w{j} (x^{j} coeff) : {weights[j]:.4f}")

    poly_terms = " + ".join([
        f"{weights[j]:.2f}*x^{j}" if j > 1 else (f"{weights[j]:.2f}*x" if j == 1 else f"{weights[j]:.2f}")
        for j in range(degree + 1)
    ])
    print(f"Fitted Curve: y = {poly_terms}")

    # Predict for new value
    pred_new = sum(weights[j] * (new_x ** j) for j in range(degree + 1))
    print("-" * 60)
    print(f"Prediction for x = {new_x}: {pred_new:.2f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
