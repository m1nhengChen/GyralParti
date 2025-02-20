import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    """
    Graph Convolutional Network (GCN) layer with optional skip connection and customizable activation.

    Attributes:
        n_channels: Output dimensionality of the layer.
        skip_connection: If True, node features are propagated without neighborhood aggregation.
        activation: Activation function to use for the final representations.
    """
    def __init__(self, input_dim, n_channels, activation='selu', skip_connection=True):
        """
        Initializes the GCN layer.

        Args:
            n_channels: Output dimensionality of the layer.
            activation: Activation function ('selu' or any callable activation).
            skip_connection: Whether to use skip connections.
        """
        super(GCN, self).__init__()
        self.n_channels = n_channels
        self.skip_connection = skip_connection
        self.activation = self._get_activation(activation)

        self.kernel = nn.Parameter(torch.empty(input_dim, n_channels))
        self.bias = nn.Parameter(torch.empty(n_channels))
        nn.init.xavier_uniform_(self.kernel)
        nn.init.zeros_(self.bias)

        if self.skip_connection:
            self.skip_weight = nn.Parameter(torch.ones(n_channels))
        else:
            self.register_parameter('skip_weight', None)

    def _get_activation(self, activation):
        if isinstance(activation, str):
            if activation.lower() == 'selu':
                return nn.SELU()
            elif activation.lower() == 'relu':
                return nn.ReLU()
            else:
                raise ValueError(f"Unsupported activation: {activation}")
        elif callable(activation):
            return activation
        else:
            raise ValueError("Activation must be a string or a callable object.")
    
    def forward(self, inputs):
        """
        Forward pass for the GCN layer.

        Args:
            inputs: A tuple containing:
                - features: Tensor of shape [n, d], node features.
                - norm_adjacency: Sparse tensor of shape [n, n], normalized adjacency matrix.

        Returns:
            Tensor of shape [n, n_channels], the node representations.
        """
        features, norm_adjacency = inputs

        # Ensures input validity
        assert isinstance(features, torch.Tensor), "Features must be a torch.Tensor."
        assert isinstance(norm_adjacency, torch.Tensor) and norm_adjacency.is_sparse, \
            "Normalized adjacency must be a sparse torch.Tensor."
        assert len(features.shape) == 2, "Features must have shape [n, d]."
        assert len(norm_adjacency.shape) == 2, "Adjacency must have shape [n, n]."
        assert features.shape[0] == norm_adjacency.shape[0], "Feature and adjacency size mismatch."

        # Linear transformation
        output = torch.matmul(features, self.kernel)

        # Skip connection
        if self.skip_connection:
            output = output * self.skip_weight + torch.sparse.mm(norm_adjacency, output)
        else:
            output = torch.sparse.mm(norm_adjacency, output)

        # Add bias
        output = output + self.bias

        # Apply activation
        return self.activation(output)
