import torch
import torch.nn as nn
import torch.nn.functional as F

class DMoN(nn.Module):
    """
    Deep Modularity Network (DMoN) layer implementation.

    Attributes:
        n_clusters: Number of clusters in the model.
        collapse_regularization: Collapse regularization weight.
        dropout_rate: Dropout rate applied to intermediate representations before softmax.
        do_unpooling: Whether to perform unpooling of the features with respect to their soft clusters.
    """

    def __init__(self, in_features, n_clusters, collapse_regularization=0.1, dropout_rate=0.0, do_unpooling=False):
        """
        Initializes the DMoN layer.

        Args:
            n_clusters: Number of clusters.
            collapse_regularization: Weight for collapse regularization.
            dropout_rate: Dropout rate applied before softmax.
            do_unpooling: If True, perform unpooling to preserve input shape.
        """
        super(DMoN, self).__init__()
        self.n_clusters = n_clusters
        self.collapse_regularization = collapse_regularization
        self.dropout_rate = dropout_rate
        self.do_unpooling = do_unpooling

        # Transormation layers
        self.transform = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=n_clusters, bias=True),
            nn.Dropout(p=dropout_rate)
            )
        
    def forward(self, features, adjacency):
        """
        Performs DMoN clustering.

        Args:
            features: Tensor of shape [n, d], node features.
            adjacency: Sparse tensor of shape [n, n], adjacency matrix.

        Returns:
            Tuple (features_pooled, assignments):
                features_pooled: Cluster representations or unpooled features.
                assignments: Soft cluster assignment matrix.
        """
        # Ensure input validity
        assert isinstance(features, torch.Tensor), "Features must be a torch.Tensor."
        assert isinstance(adjacency, torch.Tensor) and adjacency.is_sparse, "Adjacency must be a sparse tensor."
        assert len(features.shape) == 2, "Features must have shape [n, d]."
        assert len(adjacency.shape) == 2, "Adjacency must have shape [n, n]."
        assert features.shape[0] == adjacency.shape[0], "Features and adjacency size mismatch."

        # Transform features to cluster assignments
        assignments = F.softmax(self.transform(features), dim=1)

        # Cluster sizes
        cluster_sizes = torch.sum(assignments, dim=0)
        assignments_pooling = assignments / cluster_sizes.unsqueeze(0)

        # Degrees of nodes
        degrees = torch.sparse.sum(adjacency, dim=1).to_dense()
        degrees = degrees.view(-1, 1)

        # Number of edges
        number_of_nodes = adjacency.shape[1]
        number_of_edges = torch.sum(degrees)

        # Compute pooled graph: S^T * A * S
        graph_pooled = torch.sparse.mm(adjacency, assignments)
        graph_pooled = torch.mm(assignments.T, graph_pooled)

        # Compute normalizer: S^T * d * d^T * S
        normalizer_left = torch.mm(assignments.T, degrees)
        normalizer_right = torch.mm(degrees.T, assignments)
        normalizer = torch.mm(normalizer_left, normalizer_right) / (2 * number_of_edges)

        # Spectral loss
        spectral_loss = -torch.trace(graph_pooled - normalizer) / (2 * number_of_edges)

        # Collapse loss
        collapse_loss = torch.norm(cluster_sizes) / number_of_nodes * torch.sqrt(torch.tensor(float(self.n_clusters))) - 1
        total_collapse_loss = self.collapse_regularization * collapse_loss

        # Features pooling
        features_pooled = torch.mm(assignments_pooling.T, features)
        features_pooled = F.selu(features_pooled)
        if self.do_unpooling:
            features_pooled = torch.mm(assignments_pooling, features_pooled)

        # Add losses
        self.spectral_loss = spectral_loss
        self.collapse_loss = total_collapse_loss

        return features_pooled, assignments