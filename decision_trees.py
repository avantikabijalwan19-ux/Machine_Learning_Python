"""
Practical: Decision Tree Classifier from Scratch (Gini Impurity)
Description:
    Builds a decision tree split by calculating Gini Impurity across candidate thresholds.
    Gini Impurity = 1 - sum(p_i^2).
    Selects the split threshold that maximizes information gain / minimizes weighted Gini score.
"""

# Training data: [Hours_Studied, Result]
data = [
    [1, "Fail"],
    [2, "Fail"],
    [3, "Fail"],
    [6, "Pass"],
    [7, "Pass"],
    [8, "Pass"]
]

def calculate_gini(groups):
    total_samples = sum(len(g) for g in groups)
    if total_samples == 0:
        return 0.0

    gini_score = 0.0
    for group in groups:
        group_size = len(group)
        if group_size == 0:
            continue

        label_counts = {}
        for row in group:
            lbl = row[1]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        impurity = 1.0 - sum((count / group_size) ** 2 for count in label_counts.values())
        gini_score += (group_size / total_samples) * impurity

    return gini_score

def split_dataset(dataset, threshold):
    left = [row for row in dataset if row[0] < threshold]
    right = [row for row in dataset if row[0] >= threshold]
    return left, right

def find_best_split(dataset):
    unique_values = sorted(list(set(row[0] for row in dataset)))
    best_threshold = None
    best_gini = float("inf")
    best_groups = None

    print(f"{'Threshold':<15} {'Left Group':<20} {'Right Group':<20} {'Weighted Gini':<15}")
    print("-" * 70)

    for i in range(len(unique_values) - 1):
        # Candidate threshold is the midpoint between consecutive sorted values
        threshold = (unique_values[i] + unique_values[i + 1]) / 2.0
        left, right = split_dataset(dataset, threshold)
        gini = calculate_gini([left, right])

        left_str = f"{[r[0] for r in left]}"
        right_str = f"{[r[0] for r in right]}"
        print(f"{threshold:<15.2f} {left_str:<20} {right_str:<20} {gini:<15.4f}")

        if gini < best_gini:
            best_gini = gini
            best_threshold = threshold
            best_groups = (left, right)

    return best_threshold, best_gini, best_groups

def majority_class(group):
    counts = {}
    for row in group:
        counts[row[1]] = counts.get(row[1], 0) + 1
    return max(counts, key=counts.get)

def main():
    print("=" * 70)
    print("        DECISION TREE SPLITTING USING GINI IMPURITY")
    print("=" * 70)
    print("Training Dataset: [Hours, Status]")
    for row in data:
        print(f"  {row[0]} hrs -> {row[1]}")
    print()

    best_thresh, best_gini, (left, right) = find_best_split(data)

    print("-" * 70)
    print(f"Optimal Split Threshold : Hours < {best_thresh:.1f}")
    print(f"Minimum Gini Impurity   : {best_gini:.4f}")
    print(f"Left Branch Decision    : {majority_class(left)}  (Samples: {[r[0] for r in left]})")
    print(f"Right Branch Decision   : {majority_class(right)} (Samples: {[r[0] for r in right]})")
    print("=" * 70)

    # Test prediction
    test_cases = [2.5, 4.0, 7.5]
    print("\nTesting Predictions:")
    for val in test_cases:
        pred = majority_class(left) if val < best_thresh else majority_class(right)
        print(f"  Hours = {val}  -->  Predicted: {pred}")

if __name__ == "__main__":
    main()

