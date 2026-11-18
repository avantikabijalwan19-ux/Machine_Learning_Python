"""
Practical: K-Means Clustering Algorithm from Scratch
Description:
    Partitions n data points into K clusters. Each point is assigned to the nearest centroid.
    Centroids are iteratively recalculated as the arithmetic mean of points in each cluster
    until convergence.
    Accepts interactive user input for data points, K parameter, and initial centroids.
"""

import math

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
    print("           K-MEANS CLUSTERING (FROM SCRATCH)")
    print("=" * 60)

    default_data = [
        [1.0, 20.0],
        [2.0, 25.0],
        [3.0, 30.0],
        [8.0, 80.0],
        [9.0, 85.0],
        [10.0, 90.0]
    ]

    choice = input("Do you want to enter custom 2D data points? (y/n) [default: n]: ").strip().lower()
    if choice == 'y':
        try:
            n_pts = int(input("Enter number of points [default: 6]: ").strip() or "6")
            print("Enter each point as 'x y' (e.g., 2 25):")
            custom_pts = []
            for i in range(n_pts):
                parts = [float(p) for p in input(f"  Point {i + 1}: ").strip().replace(",", " ").split()]
                if len(parts) >= 2:
                    custom_pts.append(parts[:2])
            data = custom_pts if len(custom_pts) >= 2 else default_data
        except Exception as e:
            print(f"Invalid input ({e}). Using default dataset.")
            data = default_data
    else:
        data = default_data

    raw_k = input("Enter number of clusters K [default: 2]: ").strip()
    try:
        K = int(raw_k) if raw_k else 2
        K = max(1, min(K, len(data)))
    except ValueError:
        K = 2

    # Initial centroids
    print(f"\nChoose initial centroids for K = {K}:")
    default_centroids = [data[0], data[-1]] if K == 2 and len(data) >= 2 else data[:K]
    init_choice = input(f"Use default initial centroids {default_centroids}? (y/n) [default: y]: ").strip().lower()

    if init_choice == 'n':
        initial_centroids = []
        for i in range(K):
            raw_c = input(f"  Centroid {i + 1} (x y): ").strip()
            parts = [float(p) for p in raw_c.replace(",", " ").split()]
            initial_centroids.append(parts[:2])
    else:
        initial_centroids = default_centroids

    print("\n" + "-" * 60)
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
