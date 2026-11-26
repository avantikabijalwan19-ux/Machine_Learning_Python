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
    Accepts interactive user input for 2D points, eps radius, and MinPts.
"""

import math

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
                    labels[curr_neighbor] = cluster_id

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

    default_points = [
        [1.0, 1.0],
        [1.0, 2.0],
        [2.0, 1.0],
        [8.0, 8.0],
        [8.0, 9.0],
        [9.0, 8.0],
        [25.0, 25.0]
    ]

    choice = input("Do you want to enter custom 2D points? (y/n) [default: n]: ").strip().lower()
    if choice == 'y':
        try:
            n_pts = int(input("Enter number of points [default: 7]: ").strip() or "7")
            print("Enter each point as 'x y' (e.g., 1.5 2.5):")
            custom_pts = []
            for i in range(n_pts):
                raw = input(f"  Point {i + 1}: ").strip().replace(",", " ")
                parts = [float(p) for p in raw.split()]
                if len(parts) >= 2:
                    custom_pts.append(parts[:2])
            points = custom_pts if len(custom_pts) >= 1 else default_points
        except Exception as e:
            print(f"Invalid input ({e}). Using default points.")
            points = default_points
    else:
        points = default_points

    raw_eps = input("Enter epsilon radius (eps) [default: 2.0]: ").strip()
    try:
        eps = float(raw_eps) if raw_eps else 2.0
    except ValueError:
        eps = 2.0

    raw_min_pts = input("Enter MinPts [default: 2]: ").strip()
    try:
        min_pts = int(raw_min_pts) if raw_min_pts else 2
    except ValueError:
        min_pts = 2

    print("\n" + "-" * 65)
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
