"""
Practical: Image Compression and Reconstruction using PCA (Singular Value Decomposition)
Description:
    Loads a grayscale image, centers it, and applies PCA via SVD (X = U * S * V^T).
    Reconstructs the image using top K principal components.
    Calculates Mean Squared Error (MSE), Percentage of Variance Retained, and saves the visual comparison.
    Accepts user input for image path and K component list.
"""

import os
import numpy as np
from PIL import Image

def find_image_file():
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "images")
    candidates = [
        os.path.join(assets_dir, "images.jpg"),
        os.path.join(assets_dir, "iamges.jpg"),
        "images.jpg",
        "iamges.jpg",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def main():
    print("=" * 65)
    print("      IMAGE COMPRESSION & RECONSTRUCTION USING PCA (SVD)")
    print("=" * 65)

    default_img = find_image_file()
    raw_path = input(f"Enter image path [default: {default_img}]: ").strip()
    img_path = raw_path if (raw_path and os.path.exists(raw_path)) else default_img

    if not img_path or not os.path.exists(img_path):
        print("Error: Image file not found. Please provide a valid image file.")
        return

    raw_k = input("Enter K components (comma-separated) [default: 100, 50, 20, 10, 5]: ").strip()
    if raw_k:
        try:
            k_values = [int(p) for p in raw_k.replace(",", " ").split()]
        except ValueError:
            k_values = [100, 50, 20, 10, 5]
    else:
        k_values = [100, 50, 20, 10, 5]

    SIZE = 256
    img = Image.open(img_path).convert("L").resize((SIZE, SIZE))
    image = np.array(img, dtype=float)

    print(f"\nLoaded image: {img_path}")
    print(f"Resolution  : {image.shape[0]} x {image.shape[1]} pixels\n")

    # Step 1: Center image
    mean = np.mean(image, axis=0)
    X = image - mean

    # Step 2: SVD Decomposition
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    total_variance = np.sum(S ** 2)

    # Step 3: Reconstruction across various K values
    print(f"{'K Components':<15} {'MSE Loss':<15} {'Variance Retained (%)':<25} {'Compression Ratio':<20}")
    print("-" * 75)

    results = []
    for k in k_values:
        k_clamped = min(k, SIZE)
        U_k = U[:, :k_clamped]
        S_k = S[:k_clamped]
        VT_k = VT[:k_clamped, :]

        reconstructed = (U_k * S_k) @ VT_k + mean
        reconstructed = np.clip(reconstructed, 0, 255)

        mse = np.mean((image - reconstructed) ** 2)
        variance_ratio = (np.sum(S[:k_clamped] ** 2) / total_variance) * 100
        compression = (SIZE * SIZE) / (k_clamped * (2 * SIZE + 1))

        print(f"{k_clamped:<15} {mse:<15.2f} {variance_ratio:<25.2f} {compression:<20.2f}x")
        results.append((k_clamped, reconstructed, mse, variance_ratio))

    print("=" * 75)

    # Optional visual comparison plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        num_plots = len(results) + 1
        cols = 3
        rows = (num_plots + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
        axes_flat = axes.flat if hasattr(axes, 'flat') else [axes]

        axes_flat[0].imshow(image, cmap="gray", vmin=0, vmax=255)
        axes_flat[0].set_title(f"Original Image ({SIZE}x{SIZE})")

        for i, (k, recon, mse, var_pct) in enumerate(results):
            ax = axes_flat[i + 1]
            ax.imshow(recon, cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"K = {k}\nMSE = {mse:.1f} | Var = {var_pct:.1f}%")

        for ax in axes_flat:
            ax.axis("off")

        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "outputs")
        os.makedirs(output_dir, exist_ok=True)
        output_plot = os.path.join(output_dir, "pca_reconstruction_images.png")
        plt.suptitle("PCA Image Reconstruction across K Components", fontsize=15)
        plt.tight_layout()
        plt.savefig(output_plot, dpi=150)
        print(f"\nVisual comparison saved as: {output_plot}")
    except Exception as e:
        print(f"(Visual plot saving skipped: {e})")

if __name__ == "__main__":
    main()
