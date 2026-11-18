"""
Practical: Simple Linear Regression from Scratch
Description:
    Implements Ordinary Least Squares (OLS) to fit a line y = b0 + b1 * x.
    Calculates the slope (b1) and y-intercept (b0) analytically without external ML libraries.
"""

# Training dataset
x = [1, 2, 3, 4, 5]
y = [20, 30, 40, 50, 60]

print("=" * 50)
print("       SIMPLE LINEAR REGRESSION (FROM SCRATCH)")
print("=" * 50)
print("Training Data:")
for xi, yi in zip(x, y):
    print(f"  x = {xi}  -->  y = {yi}")

# Step 1: Calculate means
x_mean = sum(x) / len(x)
y_mean = sum(y) / len(y)

# Step 2: Calculate slope (b1) and intercept (b0)
numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
denominator = sum((xi - x_mean) ** 2 for xi in x)

b1 = numerator / denominator
b0 = y_mean - b1 * x_mean

print("\nLearned Parameters:")
print(f"  Slope (b1)     : {b1:.4f}")
print(f"  Intercept (b0) : {b0:.4f}")
print(f"  Regression Line: y = {b0:.2f} + {b1:.2f} * x")

# Step 3: Predict for a new value
new_x = 6
prediction = b0 + b1 * new_x
print("-" * 50)
print(f"Prediction for x = {new_x}: y = {prediction:.2f}")
print("=" * 50)

