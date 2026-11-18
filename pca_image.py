"""
Practical: Image Compression and Reconstruction using PCA (Singular Value Decomposition)
Description:
    Loads a grayscale image ('images.jpg'), centers it, and applies PCA via SVD (X = U * S * V^T).
    Reconstructs the image using top K principal components (K = 100, 50, 20, 10, 5).
    Calculates Mean Squared Error (MSE), Percentage of Variance Retained, and saves the visual comparison.
"""

import os
import numpy as np
from PIL import Image

def find_image_file():
    candidates = [
        "images.jpg",
        "iamges.jpg",
        os.path.join(os.path.dirname(__file__), "images.jpg"),
        os.path.join(os.path.dirname(__file__), "iamges.jpg"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def main():
    print("=" * 65)
    print("      IMAGE COMPRESSION & RECONSTRUCTION USING PCA (SVD)")
    print("=" * 65)

    img_path = find_image_file()

    if not img_path:
        print("Error: 'images.jpg' (or 'iamges.jpg') not found. Please provide an image file.")
        return

    SIZE = 256
    img = Image.open(img_path).convert("L").resize((SIZE, SIZE))
    image = np.array(img, dtype=float)

    print(f"Loaded image: {img_path}")
    print(f"Resolution  : {image.shape[0]} x {image.shape[1]} pixels\n")

    # Step 1: Center image
    mean = np.mean(image, axis=0)
    X = image - mean

    # Step 2: SVD Decomposition
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    total_variance = np.sum(S ** 2)

    # Step 3: Reconstruction across various K values
    k_values = [100, 50, 20, 10, 5]
    print(f"{'K Components':<15} {'MSE Loss':<15} {'Variance Retained (%)':<25} {'Compression Ratio':<20}")
    print("-" * 75)

    results = []
    for k in k_values:
        U_k = U[:, :k]
        S_k = S[:k]
        VT_k = VT[:k, :]

        reconstructed = (U_k * S_k) @ VT_k + mean
        reconstructed = np.clip(reconstructed, 0, 255)

        mse = np.mean((image - reconstructed) ** 2)
        variance_ratio = (np.sum(S[:k] ** 2) / total_variance) * 100
        compression = (SIZE * SIZE) / (k * (2 * SIZE + 1))

        print(f"{k:<15} {mse:<15.2f} {variance_ratio:<25.2f} {compression:<20.2f}x")
        results.append((k, reconstructed, mse, variance_ratio))

    print("=" * 75)

    # Optional visual comparison plot
    try:
        import matplotlib
        matplotlib.use('Agg')  # Headless-safe backend
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        axes[0, 0].imshow(image, cmap="gray", vmin=0, vmax=255)
        axes[0, 0].set_title(f"Original Image ({SIZE}x{SIZE})")

        for i, (k, recon, mse, var_pct) in enumerate(results):
            ax = axes.flat[i + 1]
            ax.imshow(recon, cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"K = {k}\nMSE = {mse:.1f} | Var = {var_pct:.1f}%")

        for ax in axes.flat:
            ax.axis("off")

        output_plot = os.path.join(os.path.dirname(__file__), "pca_reconstruction_images.png")
        plt.suptitle("PCA Image Reconstruction across K Components (images.jpg)", fontsize=15)
        plt.tight_layout()
        plt.savefig(output_plot, dpi=150)
        print(f"\nVisual comparison saved as: {output_plot}")
    except Exception as e:
        print(f"(Visual plot saving skipped: {e})")

if __name__ == "__main__":
    main()
