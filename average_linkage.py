"""
Practical: Hierarchical Agglomerative Clustering - Average Linkage
Description:
    Bottom-up clustering where clusters are merged based on Average Pairwise Distance:
    D(C1, C2) = (1 / (|C1| * |C2|)) * sum( dist(p1, p2) for p1 in C1, p2 in C2 )
    Balances the chaining tendencies of single linkage with the conservative nature of complete linkage.
    Accepts interactive user input for data points and target number of clusters K.
"""

def distance(a, b):
    return abs(a - b)

def parse_float_list(prompt_text, default_list):
    default_str = ", ".join(str(val) for val in default_list)
    raw = input(f"{prompt_text} [default: {default_str}]: ").strip()
    if not raw:
        return default_list
    try:
        parts = raw.replace(",", " ").split()
        return [float(p) for p in parts]
    except ValueError:
        print("  Invalid format. Using default values.")
        return default_list

def average_linkage_distance(c1, c2):
    total_dist = sum(distance(p1, p2) for p1 in c1 for p2 in c2)
    return total_dist / (len(c1) * len(c2))

def hierarchical_average_linkage(points, k):
    clusters = [[p] for p in points]
    step = 1

    print(f"\nInitial Clusters: {clusters}\n")

    while len(clusters) > k:
        min_dist = float("inf")
        merge_pair = (None, None)

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = average_linkage_distance(clusters[i], clusters[j])
                if d < min_dist:
                    min_dist = d
                    merge_pair = (i, j)

        i, j = merge_pair
        merged = clusters[i] + clusters[j]
        print(f"Step {step}: Merged {clusters[i]} and {clusters[j]} (Average Distance = {min_dist:.2f})")

        new_clusters = [clusters[idx] for idx in range(len(clusters)) if idx != i and idx != j]
        new_clusters.append(merged)
        clusters = new_clusters
        step += 1
        print(f"  Current Clusters: {clusters}")

    return clusters

def main():
    print("=" * 60)
    print("  HIERARCHICAL CLUSTERING: AVERAGE LINKAGE (MEAN DISTANCE)")
    print("=" * 60)

    data = parse_float_list("Enter data points", [1, 2, 8, 9])
    raw_k = input("Enter target number of clusters K [default: 2]: ").strip()
    try:
        target_k = int(raw_k) if raw_k else 2
        target_k = max(1, min(target_k, len(data)))
    except ValueError:
        target_k = 2

    print("\n" + "-" * 60)
    print(f"Data Points : {data}")
    print(f"Target K    : {target_k}")

    final_clusters = hierarchical_average_linkage(data, target_k)

    print("\n" + "=" * 60)
    print(f"Final Clusters (K={target_k}): {final_clusters}")
    print("=" * 60)

if __name__ == "__main__":
    main()
