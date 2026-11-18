"""
Practical: Decision Tree Classifier from Scratch (Gini Impurity)
Description:
    Builds a decision tree split by calculating Gini Impurity across candidate thresholds.
    Gini Impurity = 1 - sum(p_i^2).
    Selects the split threshold that maximizes information gain / minimizes weighted Gini score.
    Accepts interactive user input for training dataset and test prediction values.
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

    print(f"\n{'Threshold':<15} {'Left Group':<20} {'Right Group':<20} {'Weighted Gini':<15}")
    print("-" * 70)

    for i in range(len(unique_values) - 1):
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
    if not group:
        return "Unknown"
    counts = {}
    for row in group:
        counts[row[1]] = counts.get(row[1], 0) + 1
    return max(counts, key=counts.get)

def main():
    print("=" * 70)
    print("        DECISION TREE SPLITTING USING GINI IMPURITY")
    print("=" * 70)

    default_data = [
        [1.0, "Fail"],
        [2.0, "Fail"],
        [3.0, "Fail"],
        [6.0, "Pass"],
        [7.0, "Pass"],
        [8.0, "Pass"]
    ]

    choice = input("Do you want to enter a custom dataset? (y/n) [default: n]: ").strip().lower()
    if choice == 'y':
        try:
            n_points = int(input("Enter number of data points [default: 6]: ").strip() or "6")
            print("Enter each point as: <FeatureValue> <ClassLabel> (e.g., 2.5 Fail)")
            custom_data = []
            for i in range(n_points):
                entry = input(f"  Point {i + 1}: ").strip().split()
                if len(entry) >= 2:
                    feat_val = float(entry[0])
                    lbl_val = entry[1]
                    custom_data.append([feat_val, lbl_val])
            if len(custom_data) >= 2:
                data = custom_data
            else:
                print("Too few points entered. Using default dataset.")
                data = default_data
        except Exception as e:
            print(f"Invalid input ({e}). Using default dataset.")
            data = default_data
    else:
        data = default_data

    test_cases = parse_float_list("Enter test feature values to predict", [2.5, 4.0, 7.5])

    print("\n" + "-" * 70)
    print("Training Dataset: [Feature, Status]")
    for row in data:
        print(f"  {row[0]} -> {row[1]}")

    best_thresh, best_gini, groups = find_best_split(data)

    if best_thresh is None:
        print("\nCould not split dataset (e.g. all features have the same value).")
        return

    left, right = groups

    print("-" * 70)
    print(f"Optimal Split Threshold : Feature < {best_thresh:.2f}")
    print(f"Minimum Gini Impurity   : {best_gini:.4f}")
    print(f"Left Branch Decision    : {majority_class(left)}  (Samples: {[r[0] for r in left]})")
    print(f"Right Branch Decision   : {majority_class(right)} (Samples: {[r[0] for r in right]})")
    print("=" * 70)

    # Test prediction
    print("\nTesting Predictions:")
    for val in test_cases:
        pred = majority_class(left) if val < best_thresh else majority_class(right)
        print(f"  Feature = {val}  -->  Predicted: {pred}")
    print("=" * 70)

if __name__ == "__main__":
    main()
