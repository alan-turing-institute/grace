from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN(torch.nn.Module):
    """A graph convolutional network for subgraph classification.

    Parameters
    ----------
    input_channels : int
        The dimension of the input; i.e., length of node feature vectors
    embedding_channels : int
        The dimension of the hidden embeddings.
    node_output_classes : int
        The dimension of the node output. This is typically the number of classes in
        the classification task.
    edge_output_classes : int
        The dimension of the edge output.

    Notes
    -----
    The edge_classifier layer takes as input the concatenated features of
    the source and destination nodes of each edge; hence, its input dimension
    is equal to 2 * the number of features per node.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: List[int],
        *,
        node_output_classes: int = 2,
        edge_output_classes: int = 2,
    ):
        super(GCN, self).__init__()

        # Hidden channels start from input features:
        hidden_channels_list = [input_channels] + hidden_channels

        self.conv_layer_list = [
            GCNConv(hidden_channels_list[i], hidden_channels_list[i + 1])
            for i in range(len(hidden_channels_list) - 1)
        ]
        self.node_classifier = Linear(
            hidden_channels_list[-1], node_output_classes
        )
        self.edge_classifier = Linear(
            hidden_channels_list[-1] * 2, edge_output_classes
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        for layer in range(len(self.conv_layer_list)):
            x = self.conv_layer_list[layer](x, edge_index)
            if layer < len(self.conv_layer_list) - 1:
                x = x.relu()
            else:
                embeddings = x

        # Extract the node embeddings for feature classif:
        x = global_mean_pool(
            embeddings, batch
        )  # [batch_size, hidden_channels]

        # TODO: set dropout probability as config hyperparam:
        x = F.dropout(x, p=0.5, training=self.training)
        node_x = self.node_classifier(x)

        src, dst = edge_index
        edge_features = torch.cat(
            [embeddings[..., src, :], embeddings[..., dst, :]], axis=-1
        )

        edge_x = self.edge_classifier(edge_features)

        return node_x, edge_x
