import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    """A graph convolutional network for subgraph classification.

    Parameters
    ----------
    input_channels : int
        The dimension of the input; i.e., length of node feature vectors
    hidden_channels : int
        The dimension of the hidden embeddings.
    dropout: float
        Dropout to apply to the embeddings.
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
        hidden_channels: list[int],
        *,
        dropout: float = 0.0,
        node_output_classes: int = 2,
        edge_output_classes: int = 2,
    ):
        super(GCN, self).__init__()

        # Hidden channels start from input features:
        hidden_channels_list = [input_channels] + hidden_channels

        # Define the list of graph convolutional layers:
        self.conv_layer_list = []
        # self.conv_layer_list = [
        #     GCNConv(hidden_channels_list[i], hidden_channels_list[i + 1])
        #     for i in range(len(hidden_channels_list) - 1)
        # ]
        self.node_classifier = Linear(
            hidden_channels_list[-1], node_output_classes
        )
        self.edge_classifier = Linear(
            hidden_channels_list[-1] * 2, edge_output_classes
        )
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Perform training on input data.

        Parameters
        ----------
        x : torch.Tensor
            Node features.
        edge_index : torch.Tensor
            Edge indices.

        Returns
        -------
        node_embeddings : torch.Tensor
            Learnt features of each node after graph convolutions.
        edge_embeddings : torch.Tensor
            Concatenated embeddings of two nodes forming and edge.
        node_x : torch.Tensor
            Logit predictions of each node class.
        edge_x : torch.Tensor
            Logit predictions of each edge class.
        """
        # Ensure the model is in evaluation mode
        self.train()

        # Run through a series of graph convolutional layers:
        # for layer in range(len(self.conv_layer_list)):
        #     x = self.conv_layer_list[layer](x, edge_index)
        #     if layer < len(self.conv_layer_list) - 1:
        #         x = x.relu()
        #     else:
        #         node_embeddings = x
        node_embeddings = x

        # Implement dropout at set probability:
        # node_embeddings = F.dropout(
        #     node_embeddings, p=self.dropout, training=self.training
        # )

        node_x = self.node_classifier(node_embeddings)

        # Get the node embeddings contributing to each edge:
        src, dst = edge_index
        edge_embeddings = torch.cat(
            [node_embeddings[..., src, :], node_embeddings[..., dst, :]],
            axis=-1,
        )

        # Classify the edges through a linear layer:
        edge_x = self.edge_classifier(edge_embeddings)

        return node_embeddings, edge_embeddings, node_x, edge_x

    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Perform inference on input data.

        Parameters
        ----------
        x : torch.Tensor
            Node features.
        edge_index : torch.Tensor
            Edge indices.

        Returns
        -------
        node_embeddings : torch.Tensor
            Learnt features of each node after graph convolutions.
        edge_embeddings : torch.Tensor
            Concatenated embeddings of two nodes forming and edge.
        node_x : torch.Tensor
            Logit predictions of each node class.
        edge_x : torch.Tensor
            Logit predictions of each edge class.
        """

        # Ensure the model is in evaluation mode
        self.eval()

        # Forward pass through the model
        with torch.no_grad():
            node_emb, edge_emb, node_x, edge_x = self(x, edge_index)
            # predicted_classes = torch.argmax(node_x, dim=-1)

        return node_emb, edge_emb, node_x, edge_x
