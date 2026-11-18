"""
Practical: Hierarchical Agglomerative Clustering - Single Linkage
Description:
    Bottom-up clustering where each point starts in its own cluster.
    Clusters are iteratively merged based on Minimum Pairwise Distance (Single Linkage):
    D(C1, C2) = min { dist(p1, p2) | p1 in C1, p2 in C2 }
"""

data = [1, 2, 8, 9]
target_k = 2

def distance(a, b):
    return abs(a - b)

def single_linkage_distance(c1, c2):
    return min(distance(p1, p2) for p1 in c1 for p2 in c2)

def hierarchical_single_linkage(points, k):
    clusters = [[p] for p in points]
    step = 1

    print(f"Initial Clusters: {clusters}\n")

    while len(clusters) > k:
        min_dist = float("inf")
        merge_pair = (None, None)

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = single_linkage_distance(clusters[i], clusters[j])
                if d < min_dist:
                    min_dist = d
                    merge_pair = (i, j)

        i, j = merge_pair
        merged = clusters[i] + clusters[j]
        print(f"Step {step}: Merged {clusters[i]} and {clusters[j]} (Min Distance = {min_dist})")

        new_clusters = [clusters[idx] for idx in range(len(clusters)) if idx != i and idx != j]
        new_clusters.append(merged)
        clusters = new_clusters
        step += 1
        print(f"  Current Clusters: {clusters}")

    return clusters

def main():
    print("=" * 60)
    print("   HIERARCHICAL CLUSTERING: SINGLE LINKAGE (MIN DISTANCE)")
    print("=" * 60)
    print(f"Data Points : {data}")
    print(f"Target K    : {target_k}\n")

    final_clusters = hierarchical_single_linkage(data, target_k)

    print("\n" + "=" * 60)
    print(f"Final Clusters (K={target_k}): {final_clusters}")
    print("=" * 60)

if __name__ == "__main__":
    main()

