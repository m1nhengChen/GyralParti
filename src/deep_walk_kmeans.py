import numpy as np
import scipy.sparse
import torch
from sklearn.metrics import normalized_mutual_info_score
import utils
import metrics
import time
import os
import logging
import pandas as pd
import itertools
import warnings

# For DeepWalk + k-means, import additional modules
import networkx as nx
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
logging.getLogger('gensim').setLevel(logging.ERROR)
# Suppress specific UserWarnings from sklearn
warnings.filterwarnings(
    "ignore", 
    category=UserWarning, 
    message=".*Clustering metrics expects discrete values but received continuous values.*"
)

###############################################################################
# DeepWalk class for learning node embeddings via random walks and Word2Vec
###############################################################################
class DeepWalk:
    def __init__(self, G, embedding_dim, num_walks, walk_length, window_size):
        """
        Args:
            G (networkx.Graph): Input graph.
            embedding_dim (int): Dimension of the node embeddings.
            num_walks (int): Number of random walks per node.
            walk_length (int): Length of each random walk.
            window_size (int): Window size for Word2Vec.
        """
        self.G = G
        self.embedding_dim = embedding_dim
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.window_size = window_size

    def random_walk(self, start_node):
        """
        Perform a random walk starting from a given node.
        
        Args:
            start_node: The starting node for the random walk.
        
        Returns:
            list: A list of nodes representing the random walk.
        """
        walk = [start_node]
        while len(walk) < self.walk_length:
            current = walk[-1]
            neighbors = list(self.G.neighbors(current))
            if len(neighbors) > 0:
                walk.append(np.random.choice(neighbors))
            else:
                break
        return walk

    def learn_embeddings(self):
        """
        Execute random walks and learn node embeddings using Word2Vec.
        
        Returns:
            gensim.models.keyedvectors.KeyedVectors: Learned node embeddings.
        """
        walks = []
        nodes = list(self.G.nodes())
        for _ in range(self.num_walks):
            np.random.shuffle(nodes)
            for node in nodes:
                walk = self.random_walk(node)
                walks.append(walk)
        # Convert node ids to strings (Word2Vec expects tokens as strings)
        walks = [list(map(str, walk)) for walk in walks]
        model = Word2Vec(sentences=walks, vector_size=self.embedding_dim,
                         window=self.window_size, min_count=0, sg=1, workers=3)
        return model.wv

###############################################################################
# Utility functions
###############################################################################
def setup_logger(output_dir):
    """
    Set up logging to both a file and the console.
    
    Args:
        output_dir (str): Directory to save the log file.
    """
    log_file = os.path.join(output_dir, 'train.log')
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler()
        ]
    )

def load_npz(filename):
    """
    Load graph data from a .npz file.
    
    Args:
        filename (str): Path to the .npz file.
    
    Returns:
        tuple: (adjacency matrix (scipy.sparse.csr_matrix), features (scipy.sparse.csr_matrix))
    """
    with np.load(open(filename, 'rb'), allow_pickle=True) as loader:
        loader = dict(loader)
        adjacency = scipy.sparse.csr_matrix((loader['adj_data'],
                                               loader['adj_indices'],
                                               loader['adj_indptr']),
                                              shape=loader['adj_shape'])
        features = scipy.sparse.csr_matrix((loader['feature_data'],
                                            loader['feature_indices'],
                                            loader['feature_indptr']),
                                           shape=loader['feature_shape'])
    return adjacency, features

def scipy_to_torch_sparse(matrix):
    """
    Convert a scipy sparse matrix to a PyTorch sparse tensor.
    
    Args:
        matrix (scipy.sparse matrix): Input sparse matrix.
    
    Returns:
        torch.sparse_coo_tensor: The corresponding sparse tensor.
    """
    matrix = matrix.tocoo()
    indices = torch.from_numpy(np.array([matrix.row, matrix.col])).long()
    values = torch.from_numpy(matrix.data).float()
    shape = matrix.shape
    return torch.sparse_coo_tensor(indices, values, shape)

def compute_mean_nmi(matrix):
    """
    Compute the mean normalized mutual information (NMI) between each pair of rows.
    
    Args:
        matrix (numpy.ndarray): 2D array where each row is a set of discrete labels.
    
    Returns:
        float: Mean NMI over all row pairs.
    """
    if matrix.shape[0] < 2:
        return 0.0
    nmi_scores = [
        normalized_mutual_info_score(row1, row2)
        for row1, row2 in itertools.combinations(matrix, 2)
    ]
    return np.mean(nmi_scores)

def evaluate_clustering(adjacency, clusters):
    """
    Evaluate clustering performance using conductance and modularity metrics.
    
    Args:
        adjacency: Graph adjacency (scipy.sparse matrix).
        clusters: Cluster assignments (numpy array).
    
    Returns:
        tuple: (conductance, modularity)
    """
    conductance = metrics.conductance(adjacency, clusters)
    modularity = metrics.modularity(adjacency, clusters)
    return conductance, modularity

def save_cluster_assignments(clusters, output_path, n_clusters, subject_name, hemisphere):
    """
    Save cluster assignments to a .npy file.
    
    Args:
        clusters (numpy array): Cluster labels.
        output_path (str): Output directory.
        n_clusters (int): Number of clusters.
        subject_name (str): Subject identifier.
        hemisphere (str): 'lh' or 'rh'.
    """
    cluster_dir = os.path.join(output_path, f"clusters_{n_clusters}")
    if not os.path.exists(cluster_dir):
        os.makedirs(cluster_dir)
    file_path = os.path.join(cluster_dir, f"{subject_name}_cluster_assignments_{hemisphere}.npy")
    node_indices = np.arange(len(clusters))
    data = np.column_stack((node_indices, clusters))
    np.save(file_path, data)
    logging.info(f"Saved cluster assignments to {file_path}")

def save_cluster_features(features_torch, clusters, n_clusters, output_path, subject_name, hemisphere):
    """
    Save cluster features to a .npy file and compute mean NMI.
    
    Args:
        features_torch (torch.Tensor): Node features.
        clusters (numpy array): Cluster assignments.
        n_clusters (int): Number of clusters.
        output_path (str): Output directory.
        subject_name (str): Subject identifier.
        hemisphere (str): 'lh' or 'rh'.
    
    Returns:
        float: Mean NMI score for the cluster features.
    """
    cluster_dir = os.path.join(output_path, f"clusters_{n_clusters}")
    if not os.path.exists(cluster_dir):
        os.makedirs(cluster_dir)
    file_path = os.path.join(cluster_dir, f"{subject_name}_cluster_features_{hemisphere}.npy")
    cluster_features = []
    for cluster_id in range(n_clusters):
        cluster_indices = torch.where(torch.tensor(clusters) == cluster_id)[0]
        if len(cluster_indices) == 0:
            cluster_features.append(torch.zeros(features_torch.shape[1], device=features_torch.device))
        else:
            cluster_feature = features_torch[cluster_indices].mean(dim=0)
            cluster_features.append(cluster_feature)
    cluster_features = torch.stack(cluster_features).cpu().numpy()
    nmi_score = compute_mean_nmi(cluster_features)
    # print(nmi_score)
    cluster_ids = np.arange(n_clusters).reshape(-1, 1)
    data = np.column_stack((cluster_ids, cluster_features))
    np.save(file_path, data)
    logging.info(f"Saved cluster features to {file_path}")
    return nmi_score

###############################################################################
# process_npz_file: Process a single .npz file using DeepWalk + k-means for clustering
###############################################################################
def process_npz_file(npz_file, output_path, n_clusters, patience=50):
    """
    Process a single .npz file for clustering using DeepWalk and k-means.
    
    Args:
        npz_file (str): Path to the input .npz file.
        output_path (str): Directory where results are saved.
        n_clusters (int): Number of clusters.
        patience: (unused parameter, retained for compatibility)
    
    Returns:
        tuple: (overall conductance, overall modularity, overall mean NMI, LH time, RH time, total time)
    """
    subject_name = os.path.basename(npz_file).replace("_graph_data_lh.npz", "")
    print(npz_file)
    
    # Process LH partition
    try:
        adjacency_lh, features_lh = load_npz(npz_file)
    except Exception as e:
        logging.error(f"Error loading {npz_file}: {e}")
        return []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    try:
        _ = scipy_to_torch_sparse(adjacency_lh).to(device)
        adj_torch = scipy_to_torch_sparse(utils.normalize_graph(adjacency_lh)).to(device)
        features_torch_lh = torch.tensor(features_lh.todense(), dtype=torch.float32).to(device)
    except Exception as e:
        logging.error(f"Error converting {npz_file} to PyTorch tensors: {e}")
        return []
    
    if features_torch_lh.shape[0] == 0 or adj_torch.shape[0] == 0:
        logging.error(f"Empty graph for {npz_file}")
        return []
    
    # DeepWalk + k-means for LH partition
    try:
        start = time.time()
        G_lh = nx.from_scipy_sparse_array(adjacency_lh)
        # DeepWalk parameters
        embedding_dim = 128
        num_walks = 10
        walk_length = 80
        window_size = 10
        
        dw_lh = DeepWalk(G_lh, embedding_dim, num_walks, walk_length, window_size)
        embeddings_lh = dw_lh.learn_embeddings()
        n_nodes = adjacency_lh.shape[0]
        embedding_matrix_lh = np.zeros((n_nodes, embedding_dim))
        for i in range(n_nodes):
            embedding_matrix_lh[i, :] = embeddings_lh[str(i)]
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        clusters = kmeans.fit_predict(embedding_matrix_lh)
        end = time.time()
        lh_time = end - start
        print(f"LH running time: {lh_time:.4f}")
        
        conductance_lh, modularity_lh = evaluate_clustering(adjacency_lh, clusters)
    except Exception as e:
        logging.error(f"Error during DeepWalk+k-means for LH partition in {npz_file}: {e}")
        return []
    
    save_cluster_assignments(clusters, output_path, n_clusters, subject_name, "lh")
    MI_lh = save_cluster_features(features_torch_lh, clusters, n_clusters, output_path, subject_name, "lh")
    
    # Process RH partition
    rh_npz_file = npz_file.replace("lh", "rh")
    try:
        adjacency_rh, features_rh = load_npz(rh_npz_file)
    except Exception as e:
        logging.error(f"Error loading {rh_npz_file}: {e}")
        return []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    try:
        _ = scipy_to_torch_sparse(adjacency_rh).to(device)
        adj_torch = scipy_to_torch_sparse(utils.normalize_graph(adjacency_rh)).to(device)
        features_torch_rh = torch.tensor(features_rh.todense(), dtype=torch.float32).to(device)
    except Exception as e:
        logging.error(f"Error converting {rh_npz_file} to PyTorch tensors: {e}")
        return []
    
    if features_torch_rh.shape[0] == 0 or adj_torch.shape[0] == 0:
        logging.error(f"Empty graph for {rh_npz_file}")
        return []
    
    # DeepWalk + k-means for RH partition
    try:
        start = time.time()
        G_rh = nx.from_scipy_sparse_array(adjacency_rh)
        dw_rh = DeepWalk(G_rh, embedding_dim, num_walks, walk_length, window_size)
        embeddings_rh = dw_rh.learn_embeddings()
        n_nodes = adjacency_rh.shape[0]
        embedding_matrix_rh = np.zeros((n_nodes, embedding_dim))
        for i in range(n_nodes):
            embedding_matrix_rh[i, :] = embeddings_rh[str(i)]
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        clusters = kmeans.fit_predict(embedding_matrix_rh)
        end = time.time()
        rh_time = end - start
        print(f"RH running time: {rh_time:.4f}")
        conductance_rh, modularity_rh = evaluate_clustering(adjacency_rh, clusters)
    except Exception as e:
        logging.error(f"Error during DeepWalk+k-means for RH partition in {rh_npz_file}: {e}")
        return []
    
    save_cluster_assignments(clusters, output_path, n_clusters, subject_name, "rh")
    MI_rh = save_cluster_features(features_torch_rh, clusters, n_clusters, output_path, subject_name, "rh")
    
    overall_conductance = (conductance_lh + conductance_rh) / 2
    overall_modularity = (modularity_lh + modularity_rh) / 2
    overall_MI = (MI_rh + MI_lh) / 2
    total_time = lh_time + rh_time
    return overall_conductance, overall_modularity, overall_MI, lh_time, rh_time, total_time

###############################################################################
# Main function
###############################################################################
def main():
    npz_dir = 'HCP_feature_1056/lh'
    output_dir = 'deepwalk_results_all'
    cluster_numbers = [4,8,12,16,20,24,32,40,50]
    # cluster_numbers = [20]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    setup_logger(output_dir)
    npz_files = [os.path.join(npz_dir, f) for f in os.listdir(npz_dir) if f.endswith('.npz')]
    logging.info(f"Filtered {len(npz_files)} .npz files from {npz_dir} for processing.")
    
    summary_results = []
    conductance_results = []
    modularity_results = []
    mi_results = []
    time_results = []
    
    for n_clusters in cluster_numbers:
        for npz_file in npz_files:
            try:
                conductance, modularity, mi_score, _, _, running_time = process_npz_file(
                    npz_file, output_dir, n_clusters, patience=150
                )
                logging.info(f"File: {npz_file}, Clusters: {n_clusters}, Conductance: {conductance:.4f}, "
                             f"Modularity: {modularity:.4f}, MI: {mi_score:.4f}, Time: {running_time:.4f}")
                conductance_results.append(conductance)
                modularity_results.append(modularity)
                mi_results.append(mi_score)
                time_results.append(running_time)
            except Exception as e:
                logging.error(f"Error processing {npz_file} with {n_clusters} clusters: {e}")
        
        mean_conductance = np.mean(conductance_results) if conductance_results else 0.0
        std_conductance = np.std(conductance_results) if conductance_results else 0.0
        mean_modularity = np.mean(modularity_results) if modularity_results else 0.0
        std_modularity = np.std(modularity_results) if modularity_results else 0.0
        mean_mi = np.mean(mi_results) if mi_results else 0.0
        std_mi = np.std(mi_results) if mi_results else 0.0
        mean_time = np.mean(time_results) if time_results else 0.0
        std_time = np.std(time_results) if time_results else 0.0
        
        logging.info(f"Results for {n_clusters} clusters: Mean Conductance: {mean_conductance:.4f}, "
                     f"Std: {std_conductance:.4f}, Mean Modularity: {mean_modularity:.4f}, "
                     f"Std: {std_modularity:.4f}, Mean MI: {mean_mi:.4f}, Std: {std_mi:.4f}, "
                     f"Mean Time: {mean_time:.4f}, Std Time: {std_time:.4f}")
        
        summary_results.append({
            "Cluster Number": n_clusters,
            "Mean Conductance": mean_conductance,
            "Conductance Std": std_conductance,
            "Mean Modularity": mean_modularity,
            "Modularity Std": std_modularity,
            "Mean Mutual Information": mean_mi,
            "Mutual Information Std": std_mi,
            "Mean Time": mean_time,
            "Time Std": std_time
        })
    
    summary_csv_path = os.path.join(output_dir, "clustering_summary.csv")
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv(summary_csv_path, index=False)
    logging.info(f"Saved summary results to {summary_csv_path}")

if __name__ == '__main__':
    main()
