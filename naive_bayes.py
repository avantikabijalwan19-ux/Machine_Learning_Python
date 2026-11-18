"""
Practical: Naive Bayes Classifier from Scratch
Description:
    Implements the Naive Bayes algorithm on categorical features using Bayes' Theorem:
    P(Class | X) = P(Class) * Product( P(x_i | Class) )
    Assumes conditional independence between features given the class label.
    Accepts user input for query features and optional custom dataset.
"""

default_dataset = [
    ['Youth', 'High', 'No', 'Fair', 'No'],
    ['Youth', 'High', 'No', 'Excellent', 'No'],
    ['Middle-aged', 'High', 'No', 'Fair', 'Yes'],
    ['Senior', 'Medium', 'No', 'Fair', 'Yes'],
    ['Senior', 'Low', 'Yes', 'Fair', 'Yes'],
    ['Senior', 'Low', 'Yes', 'Excellent', 'No'],
    ['Middle-aged', 'Low', 'Yes', 'Excellent', 'Yes'],
    ['Youth', 'Medium', 'No', 'Fair', 'No'],
    ['Youth', 'Low', 'Yes', 'Fair', 'Yes'],
    ['Senior', 'Medium', 'Yes', 'Fair', 'Yes'],
    ['Youth', 'Medium', 'Yes', 'Excellent', 'Yes'],
    ['Middle-aged', 'Medium', 'No', 'Excellent', 'Yes'],
    ['Middle-aged', 'High', 'Yes', 'Fair', 'Yes'],
    ['Senior', 'Medium', 'No', 'Excellent', 'No']
]

feature_names = ["Age", "Income", "Student", "Credit"]

def calculate_posterior(data, query_features, class_label):
    class_rows = [r for r in data if r[-1] == class_label]
    if not class_rows:
        return 0.0, 0.0, [0.0] * len(query_features)

    p_class = len(class_rows) / len(data)

    conditional_probs = []
    for i, feature_val in enumerate(query_features):
        match_count = sum(1 for r in class_rows if str(r[i]).strip().lower() == str(feature_val).strip().lower())
        # Apply Laplace smoothing if count is 0 to avoid zero product
        prob = (match_count + 1) / (len(class_rows) + len(set(r[i] for r in data))) if match_count == 0 else match_count / len(class_rows)
        conditional_probs.append(prob)

    posterior = p_class
    for p in conditional_probs:
        posterior *= p

    return posterior, p_class, conditional_probs

def main():
    print("=" * 65)
    print("         NAIVE BAYES CLASSIFIER (BAYES' THEOREM)")
    print("=" * 65)

    data = default_dataset

    choice = input("Do you want to enter a custom training dataset? (y/n) [default: n]: ").strip().lower()
    if choice == 'y':
        try:
            n_rows = int(input("Enter number of rows [default: 5]: ").strip() or "5")
            print(f"Enter each row with 4 features followed by class label (e.g., Youth, High, No, Fair, No):")
            custom_data = []
            for i in range(n_rows):
                row_raw = input(f"  Row {i + 1}: ").strip()
                parts = [p.strip() for p in row_raw.replace(",", " ").split()]
                if len(parts) >= 5:
                    custom_data.append(parts[:5])
            if custom_data:
                data = custom_data
            else:
                print("No valid rows entered. Using default dataset.")
        except Exception as e:
            print(f"Invalid input ({e}). Using default dataset.")

    # User input for query sample
    print("\n--- Enter Query Sample to Classify ---")
    age = input("Enter Age (Youth / Middle-aged / Senior) [default: Youth]: ").strip() or "Youth"
    income = input("Enter Income (High / Medium / Low) [default: Medium]: ").strip() or "Medium"
    student = input("Enter Student Status (Yes / No) [default: Yes]: ").strip() or "Yes"
    credit = input("Enter Credit Rating (Fair / Excellent) [default: Fair]: ").strip() or "Fair"

    X_query = [age, income, student, credit]
    print(f"\nQuery Sample: {dict(zip(feature_names, X_query))}\n")

    # Discover unique class labels
    classes = sorted(list(set(r[-1] for r in data)))
    if len(classes) < 2:
        classes = ["Yes", "No"]

    posteriors = {}
    print("--- Detailed Conditional Probability Calculations ---")
    for cls in classes:
        prob, prior, conds = calculate_posterior(data, X_query, cls)
        posteriors[cls] = prob
        print(f"Class '{cls}':")
        print(f"  Prior P({cls}) = {prior:.4f}")
        for fname, fval, cp in zip(feature_names, X_query, conds):
            print(f"  P({fname}={fval} | {cls}) = {cp:.4f}")
        print(f"  Unnormalized P({cls} | X) = {prob:.6f}\n")

    total_prob = sum(posteriors.values())
    print("=" * 65)
    print("Normalized Posterior Probabilities:")
    for cls in classes:
        norm_pct = (posteriors[cls] / total_prob * 100) if total_prob > 0 else 0.0
        print(f"  P(Buys_Computer = {cls:<10} | X) : {norm_pct:.2f}%")

    best_class = max(posteriors, key=posteriors.get)
    print("-" * 65)
    print(f"Final Decision: {best_class}")
    print("=" * 65)

if __name__ == "__main__":
    main()
