"""
Practical: Random Forest Classifier (Bootstrap Aggregation & Ensemble Voting)
Description:
    Implements an ensemble Random Forest:
      1. Generates multiple bootstrap samples with replacement (Bagging).
      2. Trains individual decision tree stumps on each sample.
      3. Aggregates predictions across all trees via majority voting.
    Accepts interactive user input for trees, training data, and query points.
"""

import random

def calculate_gini(groups):
    total = sum(len(g) for g in groups)
    if total == 0:
        return 0.0
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        p_pass = sum(1 for r in group if r[2] == "Pass") / size
        p_fail = sum(1 for r in group if r[2] == "Fail") / size
        impurity = 1.0 - (p_pass ** 2 + p_fail ** 2)
        gini += (size / total) * impurity
    return gini

def best_stump(sample):
    best_thresh = None
    best_gini = float("inf")
    vals = sorted(list(set(r[0] for r in sample)))

    for i in range(len(vals) - 1):
        thresh = (vals[i] + vals[i + 1]) / 2.0
        left = [r for r in sample if r[0] < thresh]
        right = [r for r in sample if r[0] >= thresh]
        g = calculate_gini([left, right])
        if g < best_gini:
            best_gini = g
            best_thresh = thresh

    if best_thresh is None:
        best_thresh = vals[0]

    left = [r for r in sample if r[0] < best_thresh]
    right = [r for r in sample if r[0] >= best_thresh]

    def mode_label(grp):
        if not grp:
            return "Pass"
        passes = sum(1 for r in grp if r[2] == "Pass")
        return "Pass" if passes >= len(grp) / 2 else "Fail"

    return {
        'threshold': best_thresh,
        'left_pred': mode_label(left),
        'right_pred': mode_label(right)
    }

def predict_tree(tree, row):
    return tree['left_pred'] if row[0] < tree['threshold'] else tree['right_pred']

def main():
    print("=" * 65)
    print("        RANDOM FOREST ENSEMBLE CLASSIFIER (BAGGING)")
    print("=" * 65)
    random.seed(42)

    default_data = [
        [1.0, 50.0, "Fail"],
        [2.0, 55.0, "Fail"],
        [3.0, 60.0, "Fail"],
        [7.0, 80.0, "Pass"],
        [8.0, 85.0, "Pass"],
        [9.0, 90.0, "Pass"]
    ]

    choice = input("Do you want to enter a custom dataset? (y/n) [default: n]: ").strip().lower()
    if choice == 'y':
        try:
            n_samples = int(input("Enter number of samples [default: 6]: ").strip() or "6")
            print("Enter each sample as: <Hours> <Attendance> <Result (Pass/Fail)>:")
            custom_data = []
            for i in range(n_samples):
                parts = input(f"  Sample {i + 1}: ").strip().split()
                if len(parts) >= 3:
                    custom_data.append([float(parts[0]), float(parts[1]), parts[2]])
            data = custom_data if len(custom_data) >= 2 else default_data
        except Exception as e:
            print(f"Invalid input ({e}). Using default dataset.")
            data = default_data
    else:
        data = default_data

    raw_trees = input("Enter number of tree estimators [default: 5]: ").strip()
    n_trees = int(raw_trees) if raw_trees else 5

    # Query input
    print("\n--- Enter Test Query Point ---")
    raw_query = input("Enter test point as 'Hours Attendance' [default: 5 75]: ").strip()
    if raw_query:
        try:
            q_parts = [float(p) for p in raw_query.replace(",", " ").split()]
            test_queries = [q_parts[:2]]
        except ValueError:
            print("Invalid input. Using default test queries.")
            test_queries = [[2.0, 52.0], [5.0, 75.0], [8.0, 88.0]]
    else:
        test_queries = [[2.0, 52.0], [5.0, 75.0], [8.0, 88.0]]

    forest = []
    print(f"\nBuilding Random Forest with {n_trees} Estimators:\n")
    for t_idx in range(n_trees):
        sample = [random.choice(data) for _ in range(len(data))]
        tree = best_stump(sample)
        forest.append(tree)
        print(f"  Tree #{t_idx + 1}: Split on Feature0 < {tree['threshold']:.2f} --> "
              f"Left: {tree['left_pred']}, Right: {tree['right_pred']}")

    print("\n" + "=" * 65)
    print("Ensemble Predictions on Test Queries:")
    print("=" * 65)

    for query in test_queries:
        tree_votes = [predict_tree(t, query) for t in forest]
        pass_votes = tree_votes.count("Pass")
        fail_votes = tree_votes.count("Fail")
        final_decision = "Pass" if pass_votes >= fail_votes else "Fail"

        print(f"Query [Hours={query[0]}, Attendance={query[1]}%]:")
        print(f"  Tree Votes        : {tree_votes} (Pass: {pass_votes}, Fail: {fail_votes})")
        print(f"  Ensemble Decision : {final_decision}\n")
    print("=" * 65)

if __name__ == "__main__":
    main()
