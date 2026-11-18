"""
Practical: Naive Bayes Classifier from Scratch
Description:
    Implements the Naive Bayes algorithm on categorical features using Bayes' Theorem:
    P(Class | X) = P(Class) * Product( P(x_i | Class) )
    Assumes conditional independence between features given the class label.
"""

# Classic UCI Buy_Computer Dataset: [Age, Income, Student, Credit_Rating, Buys_Computer]
data = [
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

def calculate_posterior(query_features, class_label):
    class_rows = [r for r in data if r[4] == class_label]
    p_class = len(class_rows) / len(data)

    conditional_probs = []
    for i, feature_val in enumerate(query_features):
        match_count = sum(1 for r in class_rows if r[i] == feature_val)
        prob = match_count / len(class_rows)
        conditional_probs.append(prob)

    # Product of all conditional probabilities * prior
    posterior = p_class
    for p in conditional_probs:
        posterior *= p

    return posterior, p_class, conditional_probs

def main():
    print("=" * 65)
    print("         NAIVE BAYES CLASSIFIER (BAYES' THEOREM)")
    print("=" * 65)

    # Test Query: Youth, Medium Income, Student: Yes, Credit: Fair
    X_query = ['Youth', 'Medium', 'Yes', 'Fair']
    print(f"Query Sample: {dict(zip(feature_names, X_query))}\n")

    prob_yes, prior_yes, cond_yes = calculate_posterior(X_query, 'Yes')
    prob_no, prior_no, cond_no = calculate_posterior(X_query, 'No')

    print("--- Detailed Calculations for Class 'Yes' ---")
    print(f"  Prior P(Yes) = {prior_yes:.4f}")
    for fname, fval, cp in zip(feature_names, X_query, cond_yes):
        print(f"  P({fname}={fval} | Yes) = {cp:.4f}")
    print(f"  Unnormalized P(Yes | X) = {prob_yes:.6f}\n")

    print("--- Detailed Calculations for Class 'No' ---")
    print(f"  Prior P(No) = {prior_no:.4f}")
    for fname, fval, cp in zip(feature_names, X_query, cond_no):
        print(f"  P({fname}={fval} | No) = {cp:.4f}")
    print(f"  Unnormalized P(No | X) = {prob_no:.6f}\n")

    # Normalize probabilities
    total = prob_yes + prob_no
    p_yes_norm = (prob_yes / total) * 100
    p_no_norm = (prob_no / total) * 100

    print("=" * 65)
    print("Normalized Probabilities:")
    print(f"  P(Buys_Computer = Yes | X) : {p_yes_norm:.2f}%")
    print(f"  P(Buys_Computer = No  | X) : {p_no_norm:.2f}%")
    print("-" * 65)
    classification = "Yes (Will Buy)" if prob_yes > prob_no else "No (Will Not Buy)"
    print(f"Final Decision: {classification}")
    print("=" * 65)

if __name__ == "__main__":
    main()

