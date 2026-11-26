"""
Practical: K-Medoids Clustering Algorithm from Scratch
Description:
    Unlike K-Means where cluster centers are artificial means, K-Medoids restricts
    centroids to be actual data points (medoids). This makes it significantly more
    robust to noise and outliers.
    Accepts interactive user input for data points, K parameter, and initial medoids.
"""

def dist(a, b):
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
            if not cluster:
                continue
            best_candidate = None
            min_cost = float("inf")

            for candidate in cluster:
                cost = sum(dist(candidate, other) for other in cluster)
                if cost < min_cost:
                    min_cost = cost
                    best_candidate = candidate

            new_medoids.append(best_candidate)

        print(f"Iteration {iteration}:")
        for i in range(len(new_medoids)):
            print(f"  Cluster {i + 1} Medoid: {new_medoids[i]} | Members: {clusters[i]}")

        if new_medoids == medoids:
            print(f"\n--> Converged at iteration {iteration}!")
            break
        medoids = new_medoids

    return medoids, clusters

def main():
    print("=" * 60)
    print("           K-MEDOIDS CLUSTERING (FROM SCRATCH)")
    print("=" * 60)

    data = parse_float_list("Enter 1D data points", [2, 3, 4, 10, 11, 12])

    raw_k = input("Enter number of clusters K [default: 2]: ").strip()
    try:
        K = int(raw_k) if raw_k else 2
        K = max(1, min(K, len(data)))
    except ValueError:
        K = 2

    default_medoids = [data[0], data[-1]] if K == 2 and len(data) >= 2 else data[:K]
    raw_medoids = parse_float_list(f"Enter {K} initial medoids (must be points from data)", default_medoids)
    # Validate medoids exist in data
    medoids = [m for m in raw_medoids if m in data]
    if len(medoids) != K:
        print(f"Warning: Selected medoids not all in dataset. Using {default_medoids}")
        medoids = default_medoids

    print("\n" + "-" * 60)
    print(f"Dataset Points   : {data}")
    print(f"Initial Medoids  : {medoids}\n")

    final_medoids, final_clusters = k_medoids(data, K, medoids)

    print("\n" + "=" * 60)
    print("Final K-Medoids Results:")
    print("=" * 60)
    for i in range(len(final_medoids)):
        print(f"Cluster {i + 1}:")
        print(f"  Medoid   : {final_medoids[i]}")
        print(f"  Members  : {final_clusters[i]}")
    print("=" * 60)

if __name__ == "__main__":
    main()
