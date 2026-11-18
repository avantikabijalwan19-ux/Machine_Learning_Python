"""
Practical: K-Nearest Neighbors (KNN) for Classification and Regression
Description:
    Computes distances between a query point and all training points,
    identifies the K closest neighbors, and determines:
      - Classification: Majority vote among K neighbors.
      - Regression: Mean target value among K neighbors.
"""

def distance(a, b):
    # Euclidean distance
    if isinstance(a, (int, float)):
        return abs(a - b)
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

def knn_predict(X_train, y_train, query_point, k, task_type='classification'):
    # Calculate distance to every training sample
    distances = []
    for i in range(len(X_train)):
        d = distance(query_point, X_train[i])
        distances.append((d, y_train[i]))

    # Sort by distance
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
        avg_val = sum(val for _, val in k_nearest) / k
        return avg_val, k_nearest

def main():
    print("=" * 60)
    print("        K-NEAREST NEIGHBORS (KNN) ALGORITHM")
    print("=" * 60)

    # Dataset: Experience (Years) -> Salary (in $k) & Tier
    X_train = [1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 7.5]
    y_class = ["Low", "Low", "Mid", "Mid", "High", "High", "High"]
    y_reg   = [45.0, 52.0, 60.0, 68.0, 85.0, 110.0, 130.0]

    test_point = 2.8
    k = 3

    print(f"Test Query Point: Experience = {test_point} years, K = {k}\n")

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

