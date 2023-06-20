from typing import Tuple, Optional

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
        hidden_channels: int,
        *,
        node_output_classes: int = 2,
        edge_output_classes: int = 2,
    ):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.node_classifier = Linear(hidden_channels, node_output_classes)
        self.edge_classifier = Linear(hidden_channels * 2, edge_output_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        embeddings = self.conv3(x, edge_index)
        x = global_mean_pool(
            embeddings, batch
        )  # [batch_size, hidden_channels]

        # TODO: add if for train else eval
        x = F.dropout(x, p=0.5, training=self.training)
        node_x = self.node_classifier(x)

        src, dst = edge_index
        edge_features = torch.cat(
            [embeddings[..., src, :], embeddings[..., dst, :]], axis=-1
        )

        edge_x = self.edge_classifier(edge_features)

        return node_x, edge_x

    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ):
        # predict the labels of the subgraph, no matter the annotations (node, edge)
        pass


class Classifier:
    def __init__(self, model_type: str = "gcn", layer_list: list[int] = []):
        pass
