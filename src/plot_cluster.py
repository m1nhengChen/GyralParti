import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_matrix(matrix, title, save_path=None):
    """Plots and optionally saves a heatmap of the given matrix."""
    plt.figure(figsize=(4, 4))
    plt.imshow(matrix, cmap='magma', aspect='auto')
    plt.colorbar(label="Covariance" if "Covariance" in title else "")
    plt.xlabel("Node index")
    plt.ylabel("Node index")
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Load adjacency matrix (Example: You should replace with actual adjacency data)
adj_matrix = np.random.rand(100, 100) < 0.05  # Random sparse adjacency matrix

# Load cluster assignments
cluster_assignments = np.load("/home/minheng/UTA/graph_embedding/dmon/cluster_results/clusters_4/100206_cluster_assignments_lh.npy")[:, 1]

# Load cluster features
cluster_features = np.load("/home/minheng/UTA/graph_embedding/dmon/cluster_results/clusters_4/100206_cluster_features_lh.npy")[:, 1:]

# Compute covariance matrices
cov_matched = np.cov(cluster_features, rowvar=False)
cov_nested = np.cov(cluster_features[::-1], rowvar=False)  # Simulating nesting
cov_grouped = np.cov(np.random.permutation(cluster_features), rowvar=False)  # Simulating incomplete grouping

# Plot adjacency matrix
plot_matrix(adj_matrix, "Graph Adjacency Matrix")

# Plot feature covariance matrices
plot_matrix(cov_matched, "Feature Covariance Matrix\nMatched Clusters")
plot_matrix(cov_nested, "Feature Covariance Matrix\nNested Clusters")
plot_matrix(cov_grouped, "Feature Covariance Matrix\nGrouped Clusters")