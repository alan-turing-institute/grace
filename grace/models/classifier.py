import torch

import torch.nn.functional as F
from torch.nn import Linear, ModuleList

from torch_geometric.nn import GCNConv


class Classifier(torch.nn.Module):
    """Wrapper object to return the correct instance of the (GNN) classifier."""

    def __init__(self) -> None:
        self.models = {
            # "LINEAR" : Linear,
            "GCN": GCN,
        }

    def get_model(self, classifier_type: str, **kwargs) -> torch.nn.Module:
        classifier_type = classifier_type.upper()

        if classifier_type not in self.models:
            raise NotImplementedError(
                f"(GNN) Classifier '{classifier_type}' not implemented."
            )
        else:
            model_class = self.models[classifier_type]
            return model_class(**kwargs)


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

        # If no hidden channels are specified, train simple Linear classifier:
        if not hidden_channels:
            self.conv_layer_list = None
            self.node_classifier = Linear(input_channels, node_output_classes)
            self.edge_classifier = Linear(
                input_channels * 2, edge_output_classes
            )

        # If hidden channels are specified, create dynamic num of GCN layers:
        else:
            # Hidden channels start from input features:
            assert isinstance(hidden_channels, list)
            assert all([isinstance(num, int) for num in hidden_channels])
            hidden_channels_list = [input_channels] + hidden_channels

            self.conv_layer_list = ModuleList(
                [
                    GCNConv(
                        hidden_channels_list[i], hidden_channels_list[i + 1]
                    )
                    for i in range(len(hidden_channels_list) - 1)
                ]
            )
            self.node_classifier = Linear(
                hidden_channels_list[-1], node_output_classes
            )
            self.edge_classifier = Linear(
                hidden_channels_list[-1] * 2, edge_output_classes
            )

        # Don't forget the dropout:
        # TODO: Implement dropout at other positions, not just before Linear
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
        node_x : torch.Tensor
            Logit predictions of each node class.
        edge_x : torch.Tensor
            Logit predictions of each edge class.
        """
        # Ensure the model is in training mode:
        self.train()

        # Run through a series of graph convolutional layers:
        if self.conv_layer_list is not None:
            for layer in range(len(self.conv_layer_list)):
                x = self.conv_layer_list[layer](x, edge_index)

                # Don't perform ReLU after the last layer:
                if layer < len(self.conv_layer_list) - 1:
                    x = x.relu()

        # Rename (un)learned 'x' to node_embeddings:
        node_embeddings = x

        # Implement dropout at set probability:
        # TODO: Implement dropout at other positions
        node_embeddings = F.dropout(
            node_embeddings, p=self.dropout, training=self.training
        )

        # Classify the nodes through a linear layer:
        node_x = self.node_classifier(node_embeddings)

        # Get the node embeddings contributing to each edge:
        # TODO: Consider implementing embedding dot-product
        src, dst = edge_index
        edge_embeddings = torch.cat(
            [node_embeddings[..., src, :], node_embeddings[..., dst, :]],
            axis=-1,
        )

        # Classify the edges through a linear layer:
        edge_x = self.edge_classifier(edge_embeddings)

        return node_x, edge_x

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
        node_x : torch.Tensor
            Logit predictions of each node class.
        edge_x : torch.Tensor
            Logit predictions of each edge class.
        """

        # Ensure the model is in evaluation mode:
        self.eval()

        # Forward pass through the frozen model:
        with torch.no_grad():
            node_x, edge_x = self(x, edge_index)

        return node_x, edge_x
