"""
Practical: Simple Linear Regression from Scratch
Description:
    Implements Ordinary Least Squares (OLS) to fit a line y = b0 + b1 * x.
    Calculates the slope (b1) and y-intercept (b0) analytically without external ML libraries.
    Accepts interactive user input for training data and query points.
"""

def parse_float_list(prompt_text, default_list):
    default_str = ", ".join(str(val) for val in default_list)
    raw = input(f"{prompt_text} [default: {default_str}]: ").strip()
    if not raw:
        return default_list
    try:
        # Support space or comma separation
        parts = raw.replace(",", " ").split()
        return [float(p) for p in parts]
    except ValueError:
        print("  Invalid format. Using default values.")
        return default_list

def main():
    print("=" * 55)
    print("       SIMPLE LINEAR REGRESSION (FROM SCRATCH)")
    print("=" * 55)

    # User Input for training data
    print("--- Enter Training Dataset ---")
    x = parse_float_list("Enter x values (independent variable)", [1, 2, 3, 4, 5])
    y = parse_float_list("Enter y values (dependent variable)  ", [20, 30, 40, 50, 60])

    if len(x) != len(y) or len(x) < 2:
        print("Warning: x and y must have the same number of elements and at least 2 points.")
        print("Reverting to default dataset: x = [1, 2, 3, 4, 5], y = [20, 30, 40, 50, 60]")
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [20.0, 30.0, 40.0, 50.0, 60.0]

    raw_new_x = input("Enter new x to predict [default: 6]: ").strip()
    try:
        new_x = float(raw_new_x) if raw_new_x else 6.0
    except ValueError:
        print("  Invalid input. Using default x = 6.0")
        new_x = 6.0

    print("\n" + "-" * 55)
    print("Training Data:")
    for xi, yi in zip(x, y):
        print(f"  x = {xi}  -->  y = {yi}")

    # Step 1: Calculate means
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)

    # Step 2: Calculate slope (b1) and intercept (b0)
    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    denominator = sum((xi - x_mean) ** 2 for xi in x)

    if denominator == 0:
        print("\nError: All x values are identical. Line slope is undefined (vertical line).")
        return

    b1 = numerator / denominator
    b0 = y_mean - b1 * x_mean

    print("\nLearned Parameters:")
    print(f"  Slope (b1)     : {b1:.4f}")
    print(f"  Intercept (b0) : {b0:.4f}")
    print(f"  Regression Line: y = {b0:.2f} + {b1:.2f} * x")

    # Step 3: Predict for the new value
    prediction = b0 + b1 * new_x
    print("-" * 55)
    print(f"Prediction for x = {new_x}: y = {prediction:.2f}")
    print("=" * 55)

if __name__ == "__main__":
    main()
