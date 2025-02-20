import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import normalized_mutual_info_score
import utils
import gcn
import network
import metrics
import time
import os
import logging
import pandas as pd
import itertools
import warnings
# import npeet.lnc as lnc
# from npeet import entropy_estimators as ee
# GRAPH_PATH = 'my_data_lh/100206_graph_data.npz'
ARCHITECTURE = [128, 64]
COLLAPSE_REGULARIZATION = 0.4
DROPOUT_RATE = 0.5
N_CLUSTERS = 16
N_EPOCHS = 1500
LEARNING_RATE = 0.001

warnings.filterwarnings(
    "ignore", 
    category=UserWarning, 
    message=".*Clustering metrics expects discrete values but received*"
)


def settup_logger(output_dir):
    log_file = os.path.join(output_dir, 'train.log')
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler()
        ]
    )
# Load and process graph data
def load_npz(filename):
    """Loads a graph from a npz file"""
    with np.load(open(filename, 'rb'), allow_pickle=True) as loader:
        loader = dict(loader)
        adjacency = scipy.sparse.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']), shape=loader['adj_shape'])
        features = scipy.sparse.csr_matrix((loader['feature_data'], loader['feature_indices'], loader['feature_indptr']), shape=loader['feature_shape'])
        # print(features.shape)
        features =features[:,-75:]
        # print(features.shape)
        # label_indices = loader['label_indices']
        # labels = loader['labels']
    return adjacency, features

# Convert scipy sparce matrix to pytorch sparse tensor
def scipy_to_torch_sparse(matrix):
    """
    Converts a scipy sparse matrix to a PyTorch sparse tensor.

    Args:
        matrix: A scipy sparse matrix.

    Returns:
        A PyTorch sparse tensor.
    """
    matrix = matrix.tocoo()
    indices = torch.from_numpy(np.array([matrix.row, matrix.col])).long()
    values = torch.from_numpy(matrix.data).float()
    shape = matrix.shape
    return torch.sparse_coo_tensor(indices, values, shape)

# Graph Convolutional Network Layer
class DMoNModel(nn.Module):
    def __init__(self, input_dim, architecture, n_clusters, collapse_regularization, dropout_rate):
        """
        Builds a DMoN model.

        Args:
            input_dim: Dimension of input features.
            architecture: List of hidden dimensions for GCN layers.
            n_clusters: Number of clusters.
            collapse_regularization: Regularization for DMoN pooling.
            dropout_rate: Dropout rate for the GCN layers.
        """
        super(DMoNModel, self).__init__()
        self.gcn_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)
        self.n_clusters = n_clusters

        # Build GCN layers
        prev_dim = input_dim
        for n_channels in architecture:
            self.gcn_layers.append(gcn.GCN(prev_dim, n_channels))
            prev_dim = n_channels

        # DMoN pooling layer
        self.dmon = network.DMoN(
            in_features=prev_dim,
            n_clusters=n_clusters,
            collapse_regularization=collapse_regularization,
            dropout_rate=dropout_rate
        )
    
    def forward(self, input_features, input_graph, input_adjacency):
        """
        Forward pass for the DMoN model.

        Args:
            input_features: Tensor of shape [n, d], node features.
            input_graph: Tensor of shape [n, n], normalized graph.
            input_adjacency: Tensor of shape [n, n], graph adjacency.

        Returns:
            pool: Clustered representation.
            pool_assignment: Soft cluster assignments for nodes.
        """
        x = input_features
        for gcn_layer in self.gcn_layers:
            x = gcn_layer((x, input_graph))
            x = self.dropout(x)

        pool, pool_assignment = self.dmon(x, input_adjacency)
        return pool, pool_assignment


def compute_mean_nmi(matrix):
    """
    Computes the mean normalized mutual information (NMI) between each pair of rows 
    in an n x 219 array.

    Args:
        matrix (numpy.ndarray): A 2D array of shape (n, 219), 
                                where each row represents a set of discrete class labels.

    Returns:
        float: The mean NMI value between every pair of rows. 
               Returns 0.0 if the number of rows is less than 2.
    """
    # If there are fewer than 2 rows, directly return 0.0
    if matrix.shape[0] < 2:
        return 0.0

    # Use itertools.combinations to iterate over all unique pairs of rows
    nmi_scores = [
        normalized_mutual_info_score(row1, row2)
        for row1, row2 in itertools.combinations(matrix, 2)
    ]
    
    # Return the average NMI over all row pairs
    return np.mean(nmi_scores)


def evaluate_clustering(adjacency, clusters):
    """
    Computes metrics for clustering results.

    """
    conductance = metrics.conductance(adjacency, clusters)
    modularity = metrics.modularity(adjacency, clusters)
    # nmi = normalized_mutual_info_score(labels[label_indices], clusters[label_indices], average_method='arithmetic')

    # average_similarities = []
    # for cluster_id in range(n_clusters):
    #     cluster_indices = torch.where(torch.tensor(clusters) == cluster_id)[0]
    #     if len(cluster_indices) < 2:
    #         continue

    #     cluster_features = features_torch[cluster_indices]

    #     similarities = F.cosine_similarity(
    #         cluster_features.unsqueeze(1),
    #         cluster_features.unsqueeze(0),
    #         dim=-1
    #     )

    #     upper_tri_indices = torch.triu_indices(similarities.size(0), similarities.size(1), offset=1)
    #     mean_similarity = similarities[upper_tri_indices[0], upper_tri_indices[1]].mean().item()
    #     average_similarities.append(mean_similarity)
    
    # if len(average_similarities) == 0:
    #     score = 0.0
    # else:
    #     score = sum(average_similarities) / len(average_similarities)

    return conductance, modularity
def save_cluster_assignments_lh(clusters, output_path, n_clusters, subject_name):
    """
    Saves cluster assignments to a npy file.
    """
    cluster_dir = os.path.join(output_path, f"clusters_{n_clusters}")
    if not os.path.exists(cluster_dir):
        os.makedirs(cluster_dir)

    file_path = os.path.join(cluster_dir, f"{subject_name}_cluster_assignments_lh.npy")
    node_indices = np.arange(len(clusters))
    data = np.column_stack((node_indices, clusters))
    np.save(file_path, data)
    logging.info(f"Saved cluster assignments to {file_path}")
def save_cluster_assignments_rh(clusters, output_path, n_clusters, subject_name):
    """
    Saves cluster assignments to a npy file.
    """
    cluster_dir = os.path.join(output_path, f"clusters_{n_clusters}")
    if not os.path.exists(cluster_dir):
        os.makedirs(cluster_dir)

    file_path = os.path.join(cluster_dir, f"{subject_name}_cluster_assignments_rh.npy")
    node_indices = np.arange(len(clusters))
    data = np.column_stack((node_indices, clusters))
    np.save(file_path, data)
    logging.info(f"Saved cluster assignments to {file_path}")

def save_cluster_features_lh(features_torch, clusters, n_clusters, output_path, subject_name):
    """
    Saves cluster features to a npy file.
    """
    cluster_dir = os.path.join(output_path, f"clusters_{n_clusters}")
    if not os.path.exists(cluster_dir):
        os.makedirs(cluster_dir)

    file_path = os.path.join(cluster_dir, f"{subject_name}_cluster_features_lh.npy")
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
    # print(cluster_features.shape)
    data = np.column_stack((cluster_ids, cluster_features))
    # print(data.shape)
    np.save(file_path, data)
    logging.info(f"Saved cluster features to {file_path}")
    return nmi_score

def save_cluster_features_rh(features_torch, clusters, n_clusters, output_path, subject_name):
    """
    Saves cluster features to a npy file.
    """
    cluster_dir = os.path.join(output_path, f"clusters_{n_clusters}")
    if not os.path.exists(cluster_dir):
        os.makedirs(cluster_dir)

    file_path = os.path.join(cluster_dir, f"{subject_name}_cluster_features_rh.npy")
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
    cluster_ids = np.arange(n_clusters).reshape(-1, 1)
    data = np.column_stack((cluster_ids, cluster_features))
    np.save(file_path, data)
    logging.info(f"Saved cluster features to {file_path}")
    return nmi_score

def process_npz_file(npz_file, output_path, n_clusters, patience=50):
    """
    Process a single .npz file for a specific number of clusters.
    """
    subject_name = os.path.basename(npz_file).replace("_graph_data_lh.npz", "")
    print(npz_file)
    
    # partition for lh
    try:
        adjacency_lh, features_lh = load_npz(npz_file)
    except Exception as e:
        logging.error(f"Error loading {npz_file}: {e}")
        return []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Convert adjacency and features to PyTorch tensors
    try:
        original_adjacency = scipy_to_torch_sparse(adjacency_lh).to(device)
        adj_torch = scipy_to_torch_sparse(utils.normalize_graph(adjacency_lh)).to(device)
        features_torch_lh = torch.tensor(features_lh.todense(), dtype=torch.float32).to(device)
    except Exception as e:
        logging.error(f"Error converting {npz_file} to PyTorch tensors: {e}")
        return []

    if features_torch_lh.shape[0] == 0 or adj_torch.shape[0] == 0:
        logging.error(f"Empty graph for {npz_file}")
        return []

    feature_size = features_torch_lh.shape[1]
    results = []

    logging.info(f"Training DMoN model for {npz_file} with {n_clusters} clusters")
    model = DMoNModel(
        input_dim=feature_size,
        architecture=ARCHITECTURE,
        n_clusters=n_clusters,
        collapse_regularization=COLLAPSE_REGULARIZATION,
        dropout_rate=DROPOUT_RATE
    ).to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    best_loss = float('inf')
    patience_count = 0

    try:
        start=time.time()
        for epoch in range(N_EPOCHS):
            model.train()
            optimizer.zero_grad()

            pooled, cluster_assignments = model(features_torch_lh, original_adjacency, adj_torch)
            
            spectral_loss = model.dmon.spectral_loss
            collapse_loss = model.dmon.collapse_loss
            loss = spectral_loss + COLLAPSE_REGULARIZATION * collapse_loss

            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_count = 0
                best_model_state = model.state_dict()
            else:
                patience_count += 1

            if patience_count >= patience:
                logging.info(f"Early stopping at epoch {epoch+1} with best loss: {best_loss:.4f}")
                break

            if (epoch+1) % 10 == 0:
                logging.info(f"Epoch {epoch+1}: Loss: {loss.item():.4f}, Spectral Loss: {spectral_loss.item():.4f}, Collapse Loss: {collapse_loss.item():.4f}")
    except Exception as e:
        logging.error(f"Error during training for {npz_file}: {e}")
        return []
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logging.info(f"Loaded best model with loss: {best_loss:.4f} for evaluation.")

    # Evaluate clustering
    try:
        model.eval()
        _, assignments = model(features_torch_lh, original_adjacency, adj_torch)
        clusters = assignments.argmax(dim=1).detach().cpu().numpy()
        end=time.time()
        opt_time_lh=end-start
        print(f"lh running time:{opt_time_lh:.4f}")
        # Compute metrics
        conductance_lh, modularity_lh = evaluate_clustering(adjacency_lh, clusters)
    except Exception as e:
        logging.error(f"Error evaluating clustering for {npz_file}: {e}")
    
    save_cluster_assignments_lh(clusters, output_path, n_clusters, subject_name)
    MI_lh = save_cluster_features_lh(features_torch_lh, clusters, n_clusters, output_path, subject_name)
    
    ''' 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''
    # partition for rh
    rh_npz_file= npz_file.replace("lh", "rh")
    try:
        adjacency_rh, features_rh = load_npz(rh_npz_file)
    except Exception as e:
        logging.error(f"Error loading {rh_npz_file}: {e}")
        return []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Convert adjacency and features to PyTorch tensors
    try:
        original_adjacency = scipy_to_torch_sparse(adjacency_rh).to(device)
        adj_torch = scipy_to_torch_sparse(utils.normalize_graph(adjacency_rh)).to(device)
        features_torch_rh = torch.tensor(features_rh.todense(), dtype=torch.float32).to(device)
    except Exception as e:
        logging.error(f"Error converting {rh_npz_file} to PyTorch tensors: {e}")
        return []

    if features_torch_rh.shape[0] == 0 or adj_torch.shape[0] == 0:
        logging.error(f"Empty graph for {rh_npz_file}")
        return []

    feature_size = features_torch_rh.shape[1]
    results = []

    logging.info(f"Training DMoN model for {rh_npz_file} with {n_clusters} clusters")
    model = DMoNModel(
        input_dim=feature_size,
        architecture=ARCHITECTURE,
        n_clusters=n_clusters,
        collapse_regularization=COLLAPSE_REGULARIZATION,
        dropout_rate=DROPOUT_RATE
    ).to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    best_loss = float('inf')
    patience_count = 0
    start=time.time()
    try:
        for epoch in range(N_EPOCHS):
            model.train()
            optimizer.zero_grad()

            pooled, cluster_assignments = model(features_torch_rh, original_adjacency, adj_torch)
            
            spectral_loss = model.dmon.spectral_loss
            collapse_loss = model.dmon.collapse_loss
            loss = spectral_loss + COLLAPSE_REGULARIZATION * collapse_loss

            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_count = 0
                best_model_state = model.state_dict()
            else:
                patience_count += 1

            if patience_count >= patience:
                logging.info(f"Early stopping at epoch {epoch+1} with best loss: {best_loss:.4f}")
                break

            if (epoch+1) % 10 == 0:
                logging.info(f"Epoch {epoch+1}: Loss: {loss.item():.4f}, Spectral Loss: {spectral_loss.item():.4f}, Collapse Loss: {collapse_loss.item():.4f}")
    except Exception as e:
        logging.error(f"Error during training for {rh_npz_file}: {e}")
        return []
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logging.info(f"Loaded best model with loss: {best_loss:.4f} for evaluation.")

    # Evaluate clustering
    try:
        model.eval()
        _, assignments = model(features_torch_rh, original_adjacency, adj_torch)
        clusters = assignments.argmax(dim=1).detach().cpu().numpy()
        end=time.time()
        opt_time_rh=end-start
        print(f"rh running time:{opt_time_rh:.4f}")
        # Compute metrics
        conductance_rh, modularity_rh = evaluate_clustering(adjacency_rh, clusters)
    except Exception as e:
        logging.error(f"Error evaluating clustering for {rh_npz_file}: {e}")
    
    save_cluster_assignments_rh(clusters, output_path, n_clusters, subject_name)
    MI_rh = save_cluster_features_rh(features_torch_rh, clusters, n_clusters, output_path, subject_name)
    conductance=(conductance_lh+conductance_rh)/2
    
    modularity=(modularity_lh+modularity_rh)/2
    MI=(MI_rh+MI_lh)/2
    return conductance, modularity,MI,opt_time_lh,opt_time_rh,opt_time_rh+opt_time_lh
    
# Main function
def main():
    # npz_dir = 'my_data_lh'
    # output_dir = 'cluster_results_dicco'
    # npz_dir = 'combined_feature'
    # output_dir = 'cluster_results'
    # npz_dir = 'combined_feature_bsse'
    # output_dir = 'cluster_results_bsse_combine_4'
    npz_dir = 'HCP_feature_1056/lh'
    output_dir = 'structure_only_results'
    # cluster_numbers = list(range(10, 30))
    cluster_numbers = [4,8,12,16,20,24,32,40,50]
    # cluster_numbers = [20]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    settup_logger(output_dir)

    # subject_ids = []

    # try:
    #     subject_ids_df = pd.read_csv(csv_file_path)
    #     subject_ids = subject_ids_df.iloc[:, 0].astype(str).tolist()
    #     logging.info(f"Loaded {len(subject_ids)} subject ids from {csv_file_path}.")
    # except Exception as e:
    #     logging.error(f"Error loading subject ids from {csv_file_path}: {e}")

    # npz_files = sorted([os.path.join(npz_dir, f) for f in os.listdir(npz_dir) if f.endswith('.npz')])[:5]
    npz_files = [
        os.path.join(npz_dir, f)
        for f in os.listdir(npz_dir)
        # if f.endswith('.npz') and f.split('_')[0] in subject_ids
    ]
    logging.info(f"Filtered {len(npz_files)} .npz files from {npz_dir} for training.")
    summary_results = []
    conductance_results = []
    modularity_results = []
    mi_results = []
    time_results = []
    for cluster_number in cluster_numbers:
        for npz_file in npz_files:
            try:
                conductance, modularity,mi_score,_,_ ,running_time= process_npz_file(npz_file, output_dir, cluster_number, patience=150)
                logging.info(f"File: {npz_file}, Clusters: {cluster_number}, Conductance: {conductance:.4f}, Modularity: {modularity:.4f}, Mutual information: {mi_score:.4f}, Time: {running_time:.4f}")
                conductance_results.append(conductance)
                modularity_results.append(modularity)
                mi_results.append(mi_score)
                time_results.append(running_time)
            except Exception as e:
                logging.error(f"Error processing {npz_file} with {cluster_number} clusters: {e}")
        mean_conductance = np.mean(conductance_results) if conductance_results else 0.0
        std_conductance = np.std(conductance_results) if conductance_results else 0.0
        mean_modularity = np.mean(modularity_results) if modularity_results else 0.0
        std_modularity = np.std(modularity_results) if modularity_results else 0.0
        mean_mi = np.mean(mi_results) if mi_results else 0.0
        std_mi = np.std(mi_results) if mi_results else 0.0
        mean_time = np.mean(time_results) if time_results else 0.0
        std_time = np.std(time_results) if time_results else 0.0
        logging.info(f"Results for {cluster_number} clusters: Mean Conductance: {mean_conductance:.4f}, "
                     f"Conductance Std: {std_conductance:.4f}, Mean Modularity: {mean_modularity:.4f}, "
                     f"Modularity Std: {std_modularity:.4f}, Mean MI: {mean_mi:.4f}, MI Std: {std_mi:.4f}"
                     f"Mean Time: {mean_time:.4f}, Std Time: {std_time:.4f}",)

        summary_results.append({
            "Cluster Number": cluster_number,
            "Mean Conductance": mean_conductance,
            "Conductance Std": std_conductance,
            "Mean Modularity": mean_modularity,
            "Modularity Std": std_modularity,
            "Mean Mutual Information": mean_mi,
            "Mutual Information Std": std_mi,
            "Mean Time": mean_time,
            "Time Std": std_time
        })

    summary_csv_path = os.path.join(output_dir, f"clustering_summary.csv")
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv(summary_csv_path, index=False)
    logging.info(f"Saved summary results to {summary_csv_path}")
if __name__ == '__main__':
    main()