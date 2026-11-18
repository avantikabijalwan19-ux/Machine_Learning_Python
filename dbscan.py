"""
Practical: DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
Description:
    Clusters points based on density connectivity without requiring K upfront.
    Key parameters:
      - eps: Maximum radius of a neighborhood.
      - min_pts: Minimum number of points in the eps-neighborhood to form a dense region (Core Point).
    Labels:
      - 0: Unvisited
      - -1: Noise / Outlier
      - >= 1: Cluster identifier
"""

import math

# 2D points including clusters and an isolated outlier
points = [
    [1, 1],
    [1, 2],
    [2, 1],
    [8, 8],
    [8, 9],
    [9, 8],
    [25, 25]  # Clear Outlier / Noise
]

eps = 2.0
min_pts = 2

def euclidean_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def region_query(pts, pt_idx, radius):
    neighbors = []
    for i in range(len(pts)):
        if euclidean_dist(pts[pt_idx], pts[i]) <= radius:
            neighbors.append(i)
    return neighbors

def dbscan(pts, radius, min_samples):
    labels = [0] * len(pts)  # 0: Unvisited, -1: Noise, >0: Cluster ID
    cluster_id = 0

    for i in range(len(pts)):
        if labels[i] != 0:
            continue

        neighbors = region_query(pts, i, radius)

        if len(neighbors) < min_samples:
            labels[i] = -1  # Mark as Noise temporarily
        else:
            cluster_id += 1
            labels[i] = cluster_id

            # Expand cluster
            queue = list(neighbors)
            while queue:
                curr_neighbor = queue.pop(0)

                if labels[curr_neighbor] == -1:
                    labels[curr_neighbor] = cluster_id  # Border point salvaged from noise

                if labels[curr_neighbor] != 0:
                    continue

                labels[curr_neighbor] = cluster_id
                sub_neighbors = region_query(pts, curr_neighbor, radius)
                if len(sub_neighbors) >= min_samples:
                    queue.extend(sub_neighbors)

    return labels

def main():
    print("=" * 65)
    print("      DBSCAN DENSITY-BASED CLUSTERING FROM SCRATCH")
    print("=" * 65)
    print(f"Parameters: Epsilon (eps) = {eps}, MinPts = {min_pts}\n")

    labels = dbscan(points, eps, min_pts)

    print(f"{'Point Index':<15} {'Coordinates':<20} {'Cluster / Status':<20}")
    print("-" * 65)
    for i, (pt, lbl) in enumerate(zip(points, labels)):
        if lbl == -1:
            status = "NOISE (Outlier)"
        else:
            status = f"Cluster {lbl}"
        print(f"Point #{i + 1:<10} {str(pt):<20} {status:<20}")

    num_clusters = len(set(labels) - {-1, 0})
    num_noise = labels.count(-1)
    print("=" * 65)
    print(f"Identified Clusters : {num_clusters}")
    print(f"Identified Noise    : {num_noise} points")
    print("=" * 65)

if __name__ == "__main__":
    main()

