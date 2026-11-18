"""
Practical: Principal Component Analysis (PCA) from Scratch
Description:
    Reduces feature dimensionality while preserving maximum data variance:
      1. Centers data around the feature means.
      2. Computes the Covariance Matrix.
      3. Solves for Eigenvalues and Eigenvectors.
      4. Projects data onto the top principal components.
    Accepts interactive user input for 2D data points.
"""

import math

def main():
    print("=" * 65)
    print("      PRINCIPAL COMPONENT ANALYSIS (PCA) FROM SCRATCH")
    print("=" * 65)

    default_data = [
        [2.0, 4.0],
        [4.0, 8.0],
        [6.0, 12.0],
        [8.0, 16.0],
        [10.0, 19.0]
    ]

    choice = input("Do you want to enter custom 2D data points? (y/n) [default: n]: ").strip().lower()
    if choice == 'y':
        try:
            n_pts = int(input("Enter number of samples [default: 5]: ").strip() or "5")
            print("Enter each point as 'x y' (e.g., 2.0 4.0):")
            custom_data = []
            for i in range(n_pts):
                raw = input(f"  Sample {i + 1}: ").strip().replace(",", " ")
                parts = [float(p) for p in raw.split()]
                if len(parts) >= 2:
                    custom_data.append(parts[:2])
            if len(custom_data) >= 2:
                data = custom_data
            else:
                print("At least 2 points required. Using default data.")
                data = default_data
        except Exception as e:
            print(f"Invalid input ({e}). Using default data.")
            data = default_data
    else:
        data = default_data

    n = len(data)
    print("\n" + "-" * 65)
    print(f"Original Dataset ({n} samples, 2 dimensions):")
    for row in data:
        print(f"  {row}")

    # Step 1: Mean Centering
    mean_x = sum(r[0] for r in data) / n
    mean_y = sum(r[1] for r in data) / n

    centered = [[r[0] - mean_x, r[1] - mean_y] for r in data]
    print(f"\nFeature Means: X = {mean_x:.2f}, Y = {mean_y:.2f}")

    # Step 2: Covariance Matrix
    var_x = sum(r[0] ** 2 for r in centered) / (n - 1)
    cov_xy = sum(r[0] * r[1] for r in centered) / (n - 1)
    var_y = sum(r[1] ** 2 for r in centered) / (n - 1)

    print(f"\nCovariance Matrix:")
    print(f"  [{var_x:>8.2f}, {cov_xy:>8.2f}]")
    print(f"  [{cov_xy:>8.2f}, {var_y:>8.2f}]")

    # Step 3: Analytical Eigenvalue calculation for 2x2 matrix
    trace = var_x + var_y
    det = (var_x * var_y) - (cov_xy ** 2)

    discriminant = math.sqrt(max(0.0, trace ** 2 - 4 * det))
    lambda1 = (trace + discriminant) / 2.0
    lambda2 = (trace - discriminant) / 2.0

    print(f"\nEigenvalues:")
    print(f"  Lambda 1 (Principal): {lambda1:.4f}")
    print(f"  Lambda 2            : {lambda2:.4f}")

    # Step 4: Eigenvector for Lambda 1
    v1_raw = cov_xy
    v2_raw = lambda1 - var_x
    norm = math.sqrt(v1_raw ** 2 + v2_raw ** 2)
    if norm == 0:
        norm = 1.0
    e1 = [v1_raw / norm, v2_raw / norm]

    print(f"\nPrincipal Eigenvector (Direction of maximum variance):")
    print(f"  PC1 Vector = [{e1[0]:.4f}, {e1[1]:.4f}]")

    # Step 5: Projection of centered data onto PC1 (1D reduction)
    print("\nProjected 1D Coordinates (Transformed onto PC1):")
    for orig, c in zip(data, centered):
        pc1_val = c[0] * e1[0] + c[1] * e1[1]
        print(f"  Original: {orig}  -->  PC1: {pc1_val:>8.3f}")

    total_var = lambda1 + lambda2
    variance_explained = (lambda1 / total_var) * 100 if total_var > 0 else 0.0
    print("-" * 65)
    print(f"Explained Variance Ratio by PC1: {variance_explained:.2f}%")
    print("=" * 65)

if __name__ == "__main__":
    main()
