from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN(torch.nn.Module):
    """A graph convolutional network for subgraph classification.

    Parameters
    ----------
    input_dims : int
        The dimensions of the input.
    embedding_dims : int
        The dimensions of the hidden embeddings.
    output_dims : int
        The dimensions of the output. This is typically the number of classes in
        the classifcation task.

    """

    def __init__(
        self,
        input_dims: int,
        embedding_dims: int,
        *,
        output_dims: int = 2,
    ):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(input_dims, embedding_dims)
        self.conv2 = GCNConv(embedding_dims, embedding_dims)
        self.conv3 = GCNConv(embedding_dims, embedding_dims)
        self.linear = Linear(embedding_dims, output_dims)

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
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear(x)
        return x, embeddings
