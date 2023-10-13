import torch

import torch.nn.functional as F
from torch.nn import Linear, ModuleList

from torch_geometric.nn import GCNConv, GATv2Conv, RGCNConv

from grace.styling import LOGGER
from grace.base import EdgeProps


class Classifier(torch.nn.Module):
    """Wrapper object to return the correct (GNN) classifier instance."""

    def __init__(self) -> None:
        # self.model_types = set(["GCN", "GAT", "RGCN"])  # 'RGCN' incoming!
        self.model_types = set(["GCN", "GAT", "RGCN"])

    def get_model(self, classifier_type: str, **kwargs) -> torch.nn.Module:
        classifier_type = classifier_type.upper()

        if classifier_type not in self.model_types:
            raise NotImplementedError(
                f"(GNN) Classifier '{classifier_type}' not implemented."
            )
        else:
            return GNNModel(classifier_type, **kwargs)


class GNNModel(torch.nn.Module):
    """A graph convolutional network for subgraph classification.

    Parameters
    ----------
    classifier_type : str
        The type of the classifier to train. Important for attention returns.
    input_channels : int
        The dimension of the input; i.e., length of node feature vectors
    hidden_graph_channels : list[int],
        The dimensions of the hidden embeddings for graph convolutions.
    hidden_dense_channels : list[int],
        The dimension of the hidden embeddings for Linear (dense) layers.
    dropout: float
        Dropout to apply to the embeddings.
    node_output_classes : int
        The dimension of the node output. This is typically the number of
        classes in the classification task.
    edge_output_classes : int
        The dimension of the edge output. This is typically the number of
        classes in the classification task.
    verbose : bool
        Whether to print out the model architecture in the logger.

    Notes
    -----
    The edge_classifier layer takes as input the concatenated features of
    the source (src) & destination (dst) nodes of each edge; hence, its
    input dimension is equal to 2 * the number of features per node.
    """

    def __init__(
        self,
        classifier_type: str,
        input_channels: int,
        hidden_graph_channels: list[int],
        hidden_dense_channels: list[int],
        *,
        node_output_classes: int = 2,
        edge_output_classes: int = 2,
        num_heads: int = 1,
        dropout: float = 0.0,
        verbose: bool = False,
    ):
        super(GNNModel, self).__init__()

        # Define the model attributes:
        self.classifier_type = classifier_type
        hidden_channels_list = [input_channels]
        self.conv_layer_list = None
        self.node_dense_list = None
        self.node_classifier = None
        self.edge_classifier = None
        self.dropout = dropout

        # Define how many (if any) graph conv layers are specified:
        if hidden_graph_channels:
            hidden_channels_list.extend(hidden_graph_channels)

            if classifier_type == "GCN":
                self.conv_layer_list = ModuleList(
                    [
                        GCNConv(
                            in_channels=hidden_channels_list[i],
                            out_channels=hidden_channels_list[i + 1],
                        )
                        for i in range(len(hidden_channels_list) - 1)
                    ]
                )
            elif classifier_type == "GAT":
                self.conv_layer_list = ModuleList(
                    [
                        GATv2Conv(
                            in_channels=hidden_channels_list[i],
                            out_channels=hidden_channels_list[i + 1]
                            // num_heads,
                            heads=num_heads,
                            edge_dim=len(EdgeProps),
                            add_self_loops=True,
                            dropout=dropout,
                        )
                        for i in range(len(hidden_channels_list) - 1)
                    ]
                )
            elif classifier_type == "RGCN":
                self.conv_layer_list = ModuleList(
                    [
                        RGCNConv(
                            in_channels=hidden_channels_list[i],
                            out_channels=hidden_channels_list[i + 1],
                            num_relations=num_heads,
                        )
                        for i in range(len(hidden_channels_list) - 1)
                    ]
                )
            else:
                raise NotImplementedError(
                    f"Classifier type '{classifier_type}' not implemented."
                )

        # Consider more than just one Linear layer to squish output:
        if hidden_dense_channels:
            hidden_channels_list.extend(hidden_dense_channels)
            self.node_dense_list = ModuleList(
                [
                    Linear(
                        in_features=hidden_channels_list[
                            -len(hidden_dense_channels) + i - 1
                        ],
                        out_features=hidden_channels_list[
                            -len(hidden_dense_channels) + i
                        ],
                    )
                    for i in range(len(hidden_dense_channels))
                ]
            )
        # Final node & edge classifier:
        self.node_classifier = Linear(
            hidden_channels_list[-1], node_output_classes
        )
        self.edge_classifier = Linear(
            hidden_channels_list[-1] * 2 + len(EdgeProps), edge_output_classes
        )

        # Log the moel architecture:
        if verbose is True:
            logger_string = "Model architecture:\n"
            logger_string += f"Conv_layer_list: {self.conv_layer_list}\n"
            logger_string += f"Node_dense_list: {self.node_dense_list}\n"
            logger_string += f"Node_classifier: {self.node_classifier}\n"
            logger_string += f"Edge_classifier: {self.edge_classifier}\n"
            LOGGER.info(logger_string)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_properties: torch.Tensor,
        *,
        edge_type: torch.Tensor = None,
        edge_weight: torch.Tensor = None,
        return_attention_weights: bool = False,
    ) -> tuple[torch.Tensor]:
        """Perform training on input data.

        Parameters
        ----------
        x : torch.Tensor
            Node features.
        edge_index : torch.Tensor
            Edge indices; shape = [2, num_edges]
            where 2 are the src, dst node index.
        edge_properties : torch.Tensor
            Edge properties; shape = [num_edges, num_properties]
            where num_properties equals len(EdgeProps) dataclass

        Returns
        -------
        node_x : torch.Tensor
            Logit predictions of each node class.
        edge_x : torch.Tensor
            Logit predictions of each edge class.
        attention : torch.Tensor
            Returns tuple of torch.Tensor objects of: (
                edge_index
                    with shape = [2, num_edges + num_nodes],
                attention_weights
                    with shape = [num_edges + num_nodes, num_attention_heads]
            )
            if classifier_type == 'GAT' and return_attention_weights is True,
            otherwise attention = None.

        Notes
        -----
        - Reason for shape = num_edges + num_nodes for attention weights is
            that self-loops are added. This can be turned off with the
            'add_self_loops' parameter set to False.
        """
        # Ensure the model is in training mode:
        attention = None
        self.train()

        # Run through a series of graph convolutional layers:
        if self.conv_layer_list is not None:
            for layer in range(len(self.conv_layer_list)):
                if self.classifier_type == "GCN":
                    x = self.conv_layer_list[layer](
                        x,
                        edge_index,
                        edge_weight=edge_weight,
                    )

                elif self.classifier_type == "RGCN":
                    x = self.conv_layer_list[layer](
                        x,
                        edge_index,
                        edge_type=edge_type,
                    )

                elif self.classifier_type == "GAT":
                    if isinstance(return_attention_weights, bool):
                        x, attention = self.conv_layer_list[layer](
                            x,
                            edge_index,
                            edge_properties,
                            return_attention_weights=return_attention_weights,
                        )

                    else:
                        x = self.conv_layer_list[layer](
                            x,
                            edge_index,
                            edge_properties,
                            return_attention_weights=return_attention_weights,
                        )

                # Don't perform ReLU after the last layer:
                if layer < len(self.conv_layer_list) - 1:
                    x = x.relu()

        # Run through a series of graph convolutional layers:
        if self.node_dense_list is not None:
            for layer in range(len(self.node_dense_list)):
                x = self.node_dense_list[layer](x)

                # Don't perform ReLU after the last layer:
                if layer < len(self.node_dense_list) - 1:
                    x = x.relu()

        # Rename (un)learned 'x' to node_embeddings:
        node_embeddings = x

        # Implement dropout at set probability:
        node_embeddings = F.dropout(
            node_embeddings, p=self.dropout, training=self.training
        )

        # Classify the nodes through a linear layer:
        node_x = self.node_classifier(node_embeddings)

        # Get the node embeddings contributing to each edge:
        src, dst = edge_index
        edge_embeddings = torch.cat(
            [
                node_embeddings[..., src, :],
                node_embeddings[..., dst, :],
                edge_properties[...],
            ],
            axis=-1,
        )

        # Classify the edges through a linear layer:
        edge_x = self.edge_classifier(edge_embeddings)

        return node_x, edge_x, attention

    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_properties: torch.Tensor,
        *,
        edge_weight: torch.Tensor = None,
        return_attention_weights: bool = True,
    ) -> tuple[torch.Tensor]:
        """Perform inference on input data. Please inspect self.forward()
        method for detailed parameter documentation."""

        # Ensure the model is in evaluation mode:
        self.eval()

        # Forward pass through the frozen model:
        with torch.no_grad():
            node_x, edge_x, attention = self(
                x,
                edge_index,
                edge_properties,
                edge_weight=edge_weight,
                return_attention_weights=return_attention_weights,
            )

        return node_x, edge_x, attention
