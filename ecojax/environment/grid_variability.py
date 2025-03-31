import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.manifold import MDS
from itertools import permutations


def initialize_grid(H, W):
    """Initialize an (H, W) grid with random permutations of {0, 1, 2, 3}."""
    return np.array([np.random.permutation(4) for _ in range(H * W)]).reshape(H, W, 4)


def kendall_tau_distance(p1, p2):
    """Compute Kendall Tau distance between two permutations."""
    return sum(
        (p1[i] < p1[j]) ^ (p2[i] < p2[j])
        for i in range(len(p1))
        for j in range(i + 1, len(p1))
    )


def compute_target_permutation(neighbors, do_decrease):
    """Compute median permutation from neighbors using Borda count."""
    borda_scores = np.zeros(4)
    for p in neighbors:
        if do_decrease:
            borda_scores += np.argsort(p)
        else:
            borda_scores -= np.argsort(p)
    return np.argsort(borda_scores)


def spatial_var(grid):
    """Computes spatial variability as average element-wise differences between neighbors."""
    H, W, _ = grid.shape
    total_diff = 0
    neighbor_pairs = 0

    for i in range(H):
        for j in range(W):
            current = grid[i, j]
            if j < W - 1:
                right = grid[i, j + 1]
                total_diff += sum(1 for a, b in zip(current, right) if a != b)
                neighbor_pairs += 1
            if i < H - 1:
                bottom = grid[i + 1, j]
                total_diff += sum(1 for a, b in zip(current, bottom) if a != b)
                neighbor_pairs += 1

    return total_diff / neighbor_pairs if neighbor_pairs > 0 else 0


def adjust_permutation(current, target, diffusion_rate, do_decrease):
    """Adjust permutation to either increase or decrease similarity with target"""
    adjusted = current.copy()
    range_pos = np.random.permutation(len(current))
    
    if do_decrease:
        # DEFAULT: Decrease variability (move toward target)
        for pos in range_pos:
            if adjusted[pos] != target[pos]:
                target_pos = np.where(adjusted == target[pos])[0][0]
                adjusted[pos], adjusted[target_pos] = adjusted[target_pos], adjusted[pos]
                if np.random.rand() > diffusion_rate:
                    break
    else:
        # NEW: Increase variability (move away from target)
        for pos in range_pos:
            if adjusted[pos] == target[pos]:  # Only modify matching elements
                # Find a random position with different value to swap with
                mismatch_pos = np.where(adjusted != target)[0]
                if len(mismatch_pos) > 0:
                    swap_pos = np.random.choice(mismatch_pos)
                    adjusted[pos], adjusted[swap_pos] = adjusted[swap_pos], adjusted[pos]
                    if np.random.rand() > diffusion_rate:
                        break
    return adjusted


def add_noise(grid, noise_level=0.1):
    """Add noise to the grid by randomly swapping elements."""
    n = grid.shape[0]
    noisy_grid = grid.copy()
    for i in range(n):
        for j in range(n):
            if np.random.rand() < noise_level:
                swap_i, swap_j = np.random.randint(n), np.random.randint(n)
                noisy_grid[i, j], noisy_grid[swap_i, swap_j] = (
                    noisy_grid[swap_i, swap_j],
                    noisy_grid[i, j],
                )
    return noisy_grid


def get_grid_of_variability(var_target, H, W, diffusion_rate=0.5, iterations=1000):
    """Diffusion process with guaranteed convergence"""
    grid = initialize_grid(H, W)    
    def new_var_is_better(new_var, var):
        if var > var_target:
            return new_var < var
        else:
            if new_var == np.inf:
                return False
            return new_var > var
        
    for iteration in range(iterations):
        if iteration % 100 == 0:
            grid = initialize_grid(H, W)
        var = spatial_var(grid)
        new_grid = grid.copy()
        new_var = np.inf
        range_neighbors = 1
        while not new_var_is_better(new_var, var) and range_neighbors < H:
            for i in range(H):
                for j in range(W):
                    neighbors = [
                        grid[(i + di) % H, (j + dj) % W]
                        for di in range(-range_neighbors, range_neighbors + 1)
                        for dj in range(-range_neighbors, range_neighbors + 1)
                        if di != 0 or dj != 0
                    ]
                    target = compute_target_permutation(neighbors, do_decrease = True)
                    new_grid[i, j] = adjust_permutation(
                        grid[i, j], target, diffusion_rate, do_decrease = var > var_target,
                    )
            range_neighbors += 1
            new_var = spatial_var(new_grid)
        grid = new_grid

        if iteration % 1 == 0:
            # print(f"Iteration {iteration}: Variability = {new_var:.4f}")
            # plot_continuous_grid(grid, save_idx=iteration)
            pass

        precision = 0.01
        if abs(new_var - var_target) < precision:
            print(f"\n[INFO] Spatial var converged to var_target={var_target} (+/- {precision}) at iteration {iteration} : {new_var:.4f}")
            return grid

    raise ValueError("Diffusion did not converge after max iterations")


def plot_continuous_grid(grid, save_path=None):
    # return
    """Visualize a grid of permutations (shape n x n x 4) with MDS-based colors."""
    H, W, _ = grid.shape
    K = {0, 1, 2, 3}

    # --- Step 1: Precompute all 24 permutations of K ---
    all_perms = list(permutations(K))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}

    # --- Step 2: Compute Kendall tau distance matrix ---
    def kendall_tau(p1, p2):
        """Count pairwise inversions between two permutations."""
        return sum(
            1
            for i in range(len(p1))
            for j in range(i + 1, len(p1))
            if (p1[i] < p1[j]) != (p2[i] < p2[j])
        )

    D = np.zeros((24, 24))
    for i, p1 in enumerate(all_perms):
        for j, p2 in enumerate(all_perms):
            D[i, j] = kendall_tau(p1, p2)

    # --- Step 3: Embed permutations into 3D RGB space ---
    mds = MDS(n_components=3, dissimilarity="precomputed", random_state=42)
    embedding = mds.fit_transform(D)

    # Normalize to [0, 1] for RGB
    embedding = (embedding - embedding.min(axis=0)) / (
        embedding.max(axis=0) - embedding.min(axis=0)
    )
    perm_to_rgb = {all_perms[i]: embedding[i] for i in range(24)}

    # --- Step 4: Convert grid to RGB image ---
    rgb_image = np.zeros((H, W, 3))
    for i in range(H):
        for j in range(W):
            perm = tuple(grid[i, j])  # Assuming grid[i,j] is a permutation
            rgb_image[i, j] = perm_to_rgb[perm]

    # --- Step 5: Plot ---
    plt.imshow(rgb_image)
    plt.axis("off")
    plt.title("Permutation Grid (MDS Colors)")
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
