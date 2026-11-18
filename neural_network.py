"""
Practical: Multi-Layer Perceptron (Neural Network) with Backpropagation
Description:
    Implements a 2-layer Neural Network (2 inputs -> 2 hidden neurons -> 1 output neuron) from scratch.
    Uses Sigmoid activation and demonstrates:
      - Forward Propagation
      - Mean Squared Error Loss
      - Error Backpropagation (Chain Rule)
      - Gradient Descent Parameter Updates
    Accepts interactive user input for target function/logic gate, hyperparameters, and test queries.
"""

import math

def sigmoid(x):
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(output_val):
    return output_val * (1.0 - output_val)

def main():
    print("=" * 65)
    print("   MULTI-LAYER ARTIFICIAL NEURAL NETWORK WITH BACKPROPAGATION")
    print("=" * 65)
    print("Architecture: 2 Input Neurons --> 2 Hidden Neurons --> 1 Output Neuron")
    print("Activation  : Sigmoid | Loss: Mean Squared Error (MSE)")
    print("-" * 65)

    # Standard truth table inputs
    X = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ]

    print("Select Logic Gate / Function to Train:")
    print("  1. AND Gate  (Targets: 0, 0, 0, 1) [Default]")
    print("  2. OR Gate   (Targets: 0, 1, 1, 1)")
    print("  3. XOR Gate  (Targets: 0, 1, 1, 0)")
    print("  4. Custom 4-Value Binary Targets")
    gate_choice = input("Enter choice (1-4) [default: 1]: ").strip()

    if gate_choice == '2':
        gate_name = "OR Gate"
        Y = [0.0, 1.0, 1.0, 1.0]
    elif gate_choice == '3':
        gate_name = "XOR Gate"
        Y = [0.0, 1.0, 1.0, 0.0]
    elif gate_choice == '4':
        gate_name = "Custom Truth Table"
        raw_targets = input("Enter 4 targets for (0,0), (0,1), (1,0), (1,1) [default: 0 0 0 1]: ").strip()
        if raw_targets:
            try:
                Y = [float(p) for p in raw_targets.replace(",", " ").split()]
                if len(Y) != 4:
                    Y = [0.0, 0.0, 0.0, 1.0]
            except ValueError:
                Y = [0.0, 0.0, 0.0, 1.0]
        else:
            Y = [0.0, 0.0, 0.0, 1.0]
    else:
        gate_name = "AND Gate"
        Y = [0.0, 0.0, 0.0, 1.0]

    raw_lr = input("Enter learning rate [default: 0.5]: ").strip()
    learning_rate = float(raw_lr) if raw_lr else 0.5

    raw_epochs = input("Enter training epochs [default: 10000]: ").strip()
    epochs = int(raw_epochs) if raw_epochs else 10000

    # Initial weights
    w1, w2 = 0.5, 0.5
    w3, w4 = 0.5, 0.5
    w5, w6 = 0.5, 0.5

    b1, b2, b3 = 0.1, 0.1, 0.1

    print("\n" + "-" * 65)
    print(f"Training on: {gate_name}")
    for (x1, x2), target in zip(X, Y):
        print(f"  Input: [{int(x1)}, {int(x2)}] --> Target: {int(target)}")
    print()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for i in range(len(X)):
            x1, x2 = X[i]
            actual = Y[i]

            # ---------------------------------
            # 1. FORWARD PROPAGATION
            # ---------------------------------
            z1 = x1 * w1 + x2 * w2 + b1
            h1 = sigmoid(z1)

            z2 = x1 * w3 + x2 * w4 + b2
            h2 = sigmoid(z2)

            z3 = h1 * w5 + h2 * w6 + b3
            output = sigmoid(z3)

            # ---------------------------------
            # 2. BACKPROPAGATION
            # ---------------------------------
            error = output - actual
            total_loss += error ** 2

            out_grad = error * sigmoid_derivative(output)

            # Gradients for Hidden -> Output
            dw5 = out_grad * h1
            dw6 = out_grad * h2
            db3 = out_grad

            # Gradients for Input -> Hidden
            h1_grad = (out_grad * w5) * sigmoid_derivative(h1)
            h2_grad = (out_grad * w6) * sigmoid_derivative(h2)

            dw1 = h1_grad * x1
            dw2 = h1_grad * x2
            db1 = h1_grad

            dw3 = h2_grad * x1
            dw4 = h2_grad * x2
            db2 = h2_grad

            # ---------------------------------
            # 3. WEIGHT UPDATES
            # ---------------------------------
            w5 -= learning_rate * dw5
            w6 -= learning_rate * dw6
            b3 -= learning_rate * db3

            w1 -= learning_rate * dw1
            w2 -= learning_rate * dw2
            b1 -= learning_rate * db1

            w3 -= learning_rate * dw3
            w4 -= learning_rate * dw4
            b2 -= learning_rate * db2

        if epoch % 2000 == 0:
            mse = total_loss / len(X)
            print(f"  Epoch {epoch:>5}/{epochs} | MSE Loss: {mse:.6f}")

    print("\n" + "=" * 65)
    print("Final Predictions after Training:")
    print("=" * 65)
    print(f"{'Input (x1, x2)':<20} {'Target':<10} {'Output Prob':<15} {'Class (0/1)':<15}")
    print("-" * 65)
    for (x1, x2), target in zip(X, Y):
        h1 = sigmoid(x1 * w1 + x2 * w2 + b1)
        h2 = sigmoid(x1 * w3 + x2 * w4 + b2)
        out = sigmoid(h1 * w5 + h2 * w6 + b3)
        pred = 1 if out >= 0.5 else 0
        print(f"[{int(x1)}, {int(x2)}]{'':<14} {int(target):<10} {out:<15.4f} {pred:<15}")

    # Interactive query testing
    print("\n" + "-" * 65)
    query_raw = input("Enter custom test input (x1 x2) to predict [press Enter to skip]: ").strip()
    if query_raw:
        try:
            qx1, qx2 = [float(p) for p in query_raw.replace(",", " ").split()][:2]
            qh1 = sigmoid(qx1 * w1 + qx2 * w2 + b1)
            qh2 = sigmoid(qx1 * w3 + qx2 * w4 + b2)
            qout = sigmoid(qh1 * w5 + qh2 * w6 + b3)
            qpred = 1 if qout >= 0.5 else 0
            print(f"Prediction for [{qx1}, {qx2}]: Probability = {qout:.4f} --> Class = {qpred}")
        except Exception as e:
            print(f"Could not parse query ({e})")
    print("=" * 65)

if __name__ == "__main__":
    main()
