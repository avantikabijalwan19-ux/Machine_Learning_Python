"""
Practical: K-Nearest Neighbors (KNN) for Classification and Regression
Description:
    Computes distances between a query point and all training points,
    identifies the K closest neighbors, and determines:
      - Classification: Majority vote among K neighbors.
      - Regression: Mean target value among K neighbors.
    Accepts interactive user input for training data, K parameter, and query point.
"""

def distance(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b)
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

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

def knn_predict(X_train, y_train, query_point, k, task_type='classification'):
    distances = []
    for i in range(len(X_train)):
        d = distance(query_point, X_train[i])
        distances.append((d, y_train[i]))

    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]

    print(f"Top {k} Nearest Neighbors:")
    for d, val in k_nearest:
        print(f"  Distance: {d:.2f} | Label/Value: {val}")

    if task_type == 'classification':
        votes = {}
        for _, label in k_nearest:
            votes[label] = votes.get(label, 0) + 1
        winner = max(votes, key=votes.get)
        return winner, votes
    else:
        avg_val = sum(float(val) for _, val in k_nearest) / k
        return avg_val, k_nearest

def main():
    print("=" * 60)
    print("        K-NEAREST NEIGHBORS (KNN) ALGORITHM")
    print("=" * 60)

    default_X = [1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 7.5]
    default_class = ["Low", "Low", "Mid", "Mid", "High", "High", "High"]
    default_reg = [45.0, 52.0, 60.0, 68.0, 85.0, 110.0, 130.0]

    choice = input("Do you want to enter a custom dataset? (y/n) [default: n]: ").strip().lower()
    if choice == 'y':
        try:
            X_train = parse_float_list("Enter training X points", default_X)
            raw_c = input("Enter class labels (space/comma separated) [default: Low, Low, Mid, Mid, High, High, High]: ").strip()
            y_class = [s.strip() for s in raw_c.replace(",", " ").split()] if raw_c else default_class
            y_reg = parse_float_list("Enter regression target values", default_reg)
            if not (len(X_train) == len(y_class) == len(y_reg)):
                print("Size mismatch! Reverting to default dataset.")
                X_train, y_class, y_reg = default_X, default_class, default_reg
        except Exception as e:
            print(f"Error ({e}). Using default dataset.")
            X_train, y_class, y_reg = default_X, default_class, default_reg
    else:
        X_train, y_class, y_reg = default_X, default_class, default_reg

    # K and Query point
    raw_k = input(f"Enter value for K [default: 3, max: {len(X_train)}]: ").strip()
    try:
        k = int(raw_k) if raw_k else 3
        k = max(1, min(k, len(X_train)))
    except ValueError:
        k = 3

    raw_test = input("Enter query point (Experience in years) [default: 2.8]: ").strip()
    try:
        test_point = float(raw_test) if raw_test else 2.8
    except ValueError:
        test_point = 2.8

    print("\n" + "-" * 60)
    print(f"Test Query Point: Experience = {test_point}, K = {k}\n")

    print("--- 1. Classification (Job Tier) ---")
    predicted_class, vote_counts = knn_predict(X_train, y_class, test_point, k, 'classification')
    print(f"Vote Distribution : {vote_counts}")
    print(f"Predicted Class   : {predicted_class}\n")

    print("--- 2. Regression (Estimated Salary in $k) ---")
    predicted_salary, _ = knn_predict(X_train, y_reg, test_point, k, 'regression')
    print(f"Predicted Salary  : ${predicted_salary:.2f}k")
    print("=" * 60)

if __name__ == "__main__":
    main()
