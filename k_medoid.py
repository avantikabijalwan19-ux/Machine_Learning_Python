"""
Practical: K-Medoids Clustering Algorithm from Scratch
Description:
    Unlike K-Means where cluster centers are artificial means, K-Medoids restricts
    centroids to be actual data points (medoids). This makes it significantly more
    robust to noise and outliers.
"""

# 1D Dataset
data = [2, 3, 4, 10, 11, 12]
K = 2
initial_medoids = [2, 10]

def dist(a, b):
    return abs(a - b)

def k_medoids(points, k, medoids, max_iters=10):
    for iteration in range(1, max_iters + 1):
        clusters = [[] for _ in range(k)]

        # Step 1: Assign each point to the nearest medoid
        for pt in points:
            dists = [dist(pt, m) for m in medoids]
            nearest = dists.index(min(dists))
            clusters[nearest].append(pt)

        # Step 2: Choose new medoid for each cluster that minimizes total intra-cluster distance
        new_medoids = []
        for cluster in clusters:
            best_candidate = None
            min_cost = float("inf")

            for candidate in cluster:
                # Sum of distances from candidate to all points in cluster
                cost = sum(dist(candidate, other) for other in cluster)
                if cost < min_cost:
                    min_cost = cost
                    best_candidate = candidate

            new_medoids.append(best_candidate)

        print(f"Iteration {iteration}:")
        for i in range(k):
            print(f"  Cluster {i + 1} Medoid: {new_medoids[i]} | Members: {clusters[i]}")

        if new_medoids == medoids:
            print(f"\n--> Converged at iteration {iteration}!")
            break
        medoids = new_medoids

    return medoids, clusters

def main():
    print("=" * 60)
    print("           K-MEDOIDS CLUSTERING (K = 2)")
    print("=" * 60)
    print(f"Dataset Points   : {data}")
    print(f"Initial Medoids  : {initial_medoids}\n")

    final_medoids, final_clusters = k_medoids(data, K, initial_medoids)

    print("\n" + "=" * 60)
    print("Final K-Medoids Results:")
    print("=" * 60)
    for i in range(K):
        print(f"Cluster {i + 1}:")
        print(f"  Medoid   : {final_medoids[i]}")
        print(f"  Members  : {final_clusters[i]}")
    print("=" * 60)

if __name__ == "__main__":
    main()

