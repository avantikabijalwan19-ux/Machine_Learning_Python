"""
Practical: Random Forest Classifier (Bootstrap Aggregation & Ensemble Voting)
Description:
    Implements an ensemble Random Forest:
      1. Generates multiple bootstrap samples with replacement (Bagging).
      2. Trains individual decision tree stumps on each sample.
      3. Aggregates predictions across all trees via majority voting.
"""

import random

# Training data: [Hours_Studied, Attendance_Pct, Result]
data = [
    [1, 50, "Fail"],
    [2, 55, "Fail"],
    [3, 60, "Fail"],
    [7, 80, "Pass"],
    [8, 85, "Pass"],
    [9, 90, "Pass"]
]

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
    # Tests splits along feature 0 (Hours)
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
    random.seed(42)  # For reproducible demo

    n_trees = 5
    forest = []

    print(f"Building Random Forest with {n_trees} Estimators:\n")
    for t_idx in range(n_trees):
        # Bootstrap sample with replacement
        sample = [random.choice(data) for _ in range(len(data))]
        tree = best_stump(sample)
        forest.append(tree)
        print(f"  Tree #{t_idx + 1}: Split on Hours < {tree['threshold']:.1f} --> "
              f"Left: {tree['left_pred']}, Right: {tree['right_pred']}")

    # Test cases
    test_queries = [
        [2, 52],  # Expected Fail
        [5, 75],  # Borderline
        [8, 88]   # Expected Pass
    ]

    print("\n" + "=" * 65)
    print("Ensemble Predictions on Test Queries:")
    print("=" * 65)

    for query in test_queries:
        tree_votes = [predict_tree(t, query) for t in forest]
        pass_votes = tree_votes.count("Pass")
        fail_votes = tree_votes.count("Fail")
        final_decision = "Pass" if pass_votes > fail_votes else "Fail"

        print(f"Query [Hours={query[0]}, Attendance={query[1]}%]:")
        print(f"  Tree Votes : {tree_votes} (Pass: {pass_votes}, Fail: {fail_votes})")
        print(f"  Ensemble Decision: {final_decision}\n")

if __name__ == "__main__":
    main()

