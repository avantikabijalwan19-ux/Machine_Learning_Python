"""
Practical: K-Means Clustering Algorithm from Scratch
Description:
    Partitions n data points into K clusters. Each point is assigned to the nearest centroid.
    Centroids are iteratively recalculated as the arithmetic mean of points in each cluster
    until convergence.
"""

import math

# 2D Training Dataset
data = [
    [1, 20],
    [2, 25],
    [3, 30],
    [8, 80],
    [9, 85],
    [10, 90]
]

K = 2
initial_centroids = [
    [1, 20],
    [10, 90]
]

def euclidean_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def k_means(points, k, centroids, max_iters=10):
    for iteration in range(1, max_iters + 1):
        clusters = [[] for _ in range(k)]

        # Step 1: Assign each point to the closest centroid
        for pt in points:
            dists = [euclidean_dist(pt, c) for c in centroids]
            nearest_cluster = dists.index(min(dists))
            clusters[nearest_cluster].append(pt)

        # Step 2: Compute new centroids as the mean of cluster members
        new_centroids = []
        for c_idx, cluster in enumerate(clusters):
            if cluster:
                mean_x = sum(p[0] for p in cluster) / len(cluster)
                mean_y = sum(p[1] for p in cluster) / len(cluster)
                new_centroids.append([round(mean_x, 2), round(mean_y, 2)])
            else:
                new_centroids.append(centroids[c_idx])

        print(f"Iteration {iteration}:")
        for c_idx in range(k):
            print(f"  Cluster {c_idx + 1} Centroid: {new_centroids[c_idx]} | Points: {clusters[c_idx]}")

        # Check convergence
        if new_centroids == centroids:
            print(f"\n--> Converged after {iteration} iterations!")
            break
        centroids = new_centroids

    return centroids, clusters

def main():
    print("=" * 60)
    print("           K-MEANS CLUSTERING (K = 2)")
    print("=" * 60)
    print("Dataset Points:")
    for pt in data:
        print(f"  ({pt[0]}, {pt[1]})")
    print(f"\nInitial Centroids: {initial_centroids}\n")

    final_centroids, final_clusters = k_means(data, K, initial_centroids)

    print("\n" + "=" * 60)
    print("Final Clustering Results:")
    print("=" * 60)
    for i in range(K):
        print(f"Cluster {i + 1}:")
        print(f"  Centroid : {final_centroids[i]}")
        print(f"  Points   : {final_clusters[i]}")
    print("=" * 60)

if __name__ == "__main__":
    main()

