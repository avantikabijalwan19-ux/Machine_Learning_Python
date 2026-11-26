# Machine Learning Lab Practicals (Python Implementation)

A comprehensive, modular, and educational repository containing all core **Machine Learning Practicals** implemented from scratch in clean, idiomatic Python. This repository covers foundational Regression algorithms, Supervised Classification models, Unsupervised Clustering techniques, Dimensionality Reduction via PCA, and Artificial Neural Networks.

---

## 📚 Table of Contents
1. [Overview](#overview)
2. [Practicals Index](#practicals-index)
3. [Theoretical Foundations & Mathematical Formulations](#theoretical-foundations--mathematical-formulations)
   - [Regression Suite](#regression-suite)
   - [Classification Suite](#classification-suite)
   - [Clustering Suite](#clustering-suite)
   - [Dimensionality Reduction (PCA)](#dimensionality-reduction-pca)
   - [Artificial Neural Networks](#artificial-neural-networks)
4. [Detailed Script Descriptions](#detailed-script-descriptions)
5. [How to Run](#how-to-run)
6. [Requirements & Installation](#requirements--installation)

---

## 🔍 Overview

Machine learning models learn patterns directly from data. This practical collection is designed to build intuition by writing the algorithms **from mathematical first principles** (avoiding black-box wrappers where possible) while providing production-ready demonstrations:

- **Supervised Learning:**
  - *Regression:* Simple Linear, Multiple Linear, Polynomial.
  - *Classification:* Logistic Regression, K-Nearest Neighbors (KNN), Decision Trees (Gini Impurity), Random Forest (Bagging), Naive Bayes (Bayes' Theorem).
- **Unsupervised Learning:**
  - *Partition-Based:* K-Means, K-Medoids (PAM).
  - *Hierarchical Agglomerative:* Single Linkage, Complete Linkage, Average Linkage.
  - *Density-Based:* DBSCAN.
- **Dimensionality Reduction:**
  - 2D Analytical PCA (Eigenvalues/Eigenvectors).
  - Image Compression using Singular Value Decomposition (SVD).
- **Deep Learning:**
  - Multi-Layer Perceptron (2-2-1) with Forward Pass, Backpropagation, and Gradient Descent.

## 📁 Repository Structure

- `supervised_learning/regression/`: Linear, multiple, and polynomial regression.
- `supervised_learning/classification/`: Logistic regression, KNN, decision trees, and Naive Bayes.
- `supervised_learning/ensemble/`: Random forest.
- `unsupervised_learning/clustering/`: Partition-based, hierarchical, and density-based clustering.
- `dimensionality_reduction/`: PCA examples, including image compression.
- `deep_learning/`: Neural network practical.
- `assets/images/`: Source images used by practicals.
- `assets/outputs/`: Generated and example output plots.

---

## 🗂 Practicals Index

| File Name | Domain | Technique / Core Idea |
|---|---|---|
| `supervised_learning/regression/linear_regression.py` | Supervised - Regression | Ordinary Least Squares (OLS) closed-form formula for line slope and intercept. |
| `supervised_learning/regression/multiple_regression.py` | Supervised - Regression | Batch Gradient Descent optimization across multi-feature datasets. |
| `supervised_learning/regression/polynomial_regression.py` | Supervised - Regression | Fits non-linear polynomial curves $y = \sum w_j x^j$ using gradient descent. |
| `supervised_learning/classification/logistic_regression.py` | Supervised - Classification | Binary classification via Sigmoid activation and cross-entropy gradient updates. |
| `supervised_learning/classification/knn.py` | Supervised - Both | Instance-based learning for classification (majority vote) and regression (mean). |
| `supervised_learning/classification/decision_trees.py` | Supervised - Classification | Evaluates candidate split thresholds using weighted Gini Impurity. |
| `supervised_learning/ensemble/random_forest.py` | Supervised - Ensemble | Bootstrap sampling with replacement (Bagging) and majority voting ensemble. |
| `supervised_learning/classification/naive_bayes.py` | Supervised - Classification | Evaluates class posterior probabilities assuming conditional feature independence. |
| `unsupervised_learning/clustering/k_means.py` | Unsupervised - Clustering | Iterative centroid assignment and cluster mean updates. |
| `unsupervised_learning/clustering/k_medoid.py` | Unsupervised - Clustering | Centroids restricted to actual dataset members (robust to outliers). |
| `unsupervised_learning/clustering/single_linkage.py` | Unsupervised - Hierarchical | Merges clusters based on minimum pairwise distance between members. |
| `unsupervised_learning/clustering/complete_linkage.py` | Unsupervised - Hierarchical | Merges clusters based on maximum pairwise distance between members. |
| `unsupervised_learning/clustering/average_linkage.py` | Unsupervised - Hierarchical | Merges clusters based on mean pairwise distance between members. |
| `unsupervised_learning/clustering/dbscan.py` | Unsupervised - Clustering | Density-based spatial clustering identifying core points, border points, and noise. |
| `dimensionality_reduction/pca.py` | Dimensionality Reduction | Mean centering, covariance matrix, analytical eigenvalues/eigenvectors. |
| `dimensionality_reduction/pca_image.py` | Dimensionality Reduction | SVD-based grayscale image compression ($K = 100, 50, 20, 10, 5$). |
| `deep_learning/neural_network.py` | Deep Learning | 2-Layer Neural Network with Sigmoid activation, MSE loss, and Backpropagation. |

---

## 📐 Theoretical Foundations & Mathematical Formulations

### Regression Suite
- **Simple Linear Regression:**
  $$y = b_0 + b_1 x \quad , \quad b_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2} \quad , \quad b_0 = \bar{y} - b_1 \bar{x}$$
- **Multiple Linear Regression:**
  $$\hat{y} = b + \sum_{j=1}^m w_j x_j \quad , \quad w_j \leftarrow w_j - \alpha \frac{\partial J}{\partial w_j}$$
- **Polynomial Regression:**
  $$\hat{y} = \sum_{j=0}^d w_j x^j$$

### Classification Suite
- **Logistic Regression:**
  $$\sigma(z) = \frac{1}{1 + e^{-z}} \quad \text{where} \quad z = w \cdot x + b$$
- **Decision Trees (Gini Impurity):**
  $$Gini = 1 - \sum_{i=1}^C p_i^2 \quad , \quad Gini_{split} = \frac{|D_1|}{|D|} Gini(D_1) + \frac{|D_2|}{|D|} Gini(D_2)$$
- **Naive Bayes:**
  $$P(C_k \mid X) \propto P(C_k) \prod_{i=1}^n P(x_i \mid C_k)$$

### Clustering Suite
- **K-Means Objective:**
  $$J = \sum_{k=1}^K \sum_{x \in S_k} \| x - \mu_k \|^2$$
- **Linkage Distances:**
  - *Single Linkage:* $D(A, B) = \min \{ d(x, y) : x \in A, y \in B \}$
  - *Complete Linkage:* $D(A, B) = \max \{ d(x, y) : x \in A, y \in B \}$
  - *Average Linkage:* $D(A, B) = \frac{1}{|A||B|} \sum_{x \in A} \sum_{y \in B} d(x, y)$

### Dimensionality Reduction (PCA)
- Mean centering: $X_c = X - \mu$
- Covariance: $\Sigma = \frac{1}{n-1} X_c^T X_c$
- Eigenvalue Decomposition: $\Sigma v = \lambda v$
- Projection: $Z = X_c V_k$

### Artificial Neural Networks
- **Forward Pass:** $z = W x + b \implies a = \sigma(z)$
- **Output Error Gradient:** $\delta_{out} = (\hat{y} - y) \cdot \sigma'(z_{out})$
- **Hidden Error Gradient:** $\delta_{hidden} = (W_{out}^T \delta_{out}) \cdot \sigma'(z_{hidden})$
- **Weight Update:** $W \leftarrow W - \eta (\delta \cdot a^T)$

---

## 📖 Detailed Script Descriptions

1. **`supervised_learning/regression/linear_regression.py`**: Computes best-fit line parameters analytically using closed-form least-squares equations.
2. **`supervised_learning/regression/multiple_regression.py`**: Optimizes multi-variable weights and bias via batch gradient descent.
3. **`supervised_learning/regression/polynomial_regression.py`**: Models non-linear data distributions with configurable polynomial degrees.
4. **`supervised_learning/classification/logistic_regression.py`**: Demonstrates probabilistic binary classification using the sigmoid activation function.
5. **`supervised_learning/classification/knn.py`**: Classifies job levels and estimates salaries based on proximity to nearest neighbors.
6. **`supervised_learning/classification/decision_trees.py`**: Evaluates all candidate splits across hours studied to find the minimum Gini impurity.
7. **`supervised_learning/ensemble/random_forest.py`**: Implements ensemble bootstrap sampling (bagging) with 5 independent tree estimators and majority voting.
8. **`supervised_learning/classification/naive_bayes.py`**: Evaluates whether a customer buys a computer using historical conditional probabilities.
9. **`unsupervised_learning/clustering/k_means.py`**: Iteratively partitions 2D data points into $K=2$ clusters with centroid convergence tracking.
10. **`unsupervised_learning/clustering/k_medoid.py`**: Groups data around actual dataset members, minimizing intra-cluster absolute distances.
11. **`unsupervised_learning/clustering/single_linkage.py`**: Step-by-step agglomerative clustering merging clusters with shortest gap.
12. **`unsupervised_learning/clustering/complete_linkage.py`**: Agglomerative clustering merging clusters with smallest maximum diameter.
13. **`unsupervised_learning/clustering/average_linkage.py`**: Agglomerative clustering merging clusters with lowest mean inter-point distance.
14. **`unsupervised_learning/clustering/dbscan.py`**: Density clustering with $\epsilon$ and $MinPts$, identifying clusters and isolating noise points.
15. **`dimensionality_reduction/pca.py`**: 2D feature projection calculating covariance matrices, eigenvalues, and principal vectors from scratch.
16. **`dimensionality_reduction/pca_image.py`**: Real-world SVD image reconstruction of `assets/images/images.jpg` across $K = 100, 50, 20, 10, 5$ components, calculating MSE, compression ratio, and variance retained. Output saved as `assets/outputs/pca_reconstruction_images.png`.

| K Components | MSE Loss | Variance Retained | Compression Ratio |
|---|---|---|---|
| **K = 100** | 4.35 | 99.89% | 1.28x |
| **K = 50** | 35.46 | 99.07% | 2.56x |
| **K = 20** | 148.13 | 96.08% | 6.39x |
| **K = 10** | 303.47 | 91.98% | 12.78x |
| **K = 5** | 544.32 | 85.50% | 25.55x |
17. **`deep_learning/neural_network.py`**: Two-layer backpropagation neural network trained on boolean logic gates.

---

## 🚀 How to Run

Execute any script directly in Python 3. Each script provides interactive prompts for user input (custom training data, queries, and parameters), while also offering sensible defaults if you press `Enter`:

```bash
# 1. Regression
python supervised_learning/regression/linear_regression.py
python supervised_learning/regression/multiple_regression.py
python supervised_learning/regression/polynomial_regression.py

# 2. Classification
python supervised_learning/classification/logistic_regression.py
python supervised_learning/classification/knn.py
python supervised_learning/classification/decision_trees.py
python supervised_learning/ensemble/random_forest.py
python supervised_learning/classification/naive_bayes.py

# 3. Clustering
python unsupervised_learning/clustering/k_means.py
python unsupervised_learning/clustering/k_medoid.py
python unsupervised_learning/clustering/single_linkage.py
python unsupervised_learning/clustering/complete_linkage.py
python unsupervised_learning/clustering/average_linkage.py
python unsupervised_learning/clustering/dbscan.py

# 4. Dimensionality Reduction
python dimensionality_reduction/pca.py
python dimensionality_reduction/pca_image.py

# 5. Neural Networks
python deep_learning/neural_network.py
```

---

## ⚙️ Requirements & Installation

- **Python Version:** Python 3.8+ (tested on Python 3.10)
- **Standard Library:** Almost all scripts run entirely with standard Python (`math`, `random`, `collections`, `os`).
- **Optional Packages (for `pca_image.py`):**
  ```bash
  pip install numpy pillow matplotlib
  ```

