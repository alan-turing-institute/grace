import torch

import numpy as np
import networkx as nx
import numpy.typing as npt

from grace.base import GraphAttrs


class LaplacianEmbedder:
    def __init__(self, graph: nx.graph) -> None:
        self.graph = graph
        self.laplacian = None

    def calculate_graph_laplacian(self) -> None:
        self.laplacian = nx.laplacian_matrix(self.graph).toarray()

    def extract_node_features(self) -> npt.NDArray:
        feature_matrix = np.stack(
            [
                n[GraphAttrs.NODE_IMG_EMBEDDING]
                for _, n in self.graph.nodes(data=True)
            ],
            axis=0,
        )
        return np.transpose(feature_matrix)

    def transform_feature_embeddings(self) -> nx.Graph:
        # Calculate the transposed feature matrix:
        self.calculate_graph_laplacian()
        feature_matrix = self.extract_node_features()

        # Perform the calculation:
        embedded_matrix = np.matmul(feature_matrix, self.laplacian)

        # Append node attributes to the graph:
        for node_idx, node in self.graph.nodes(data=True):
            node[GraphAttrs.NODE_ENV_EMBEDDING] = torch.Tensor(
                embedded_matrix[:, node_idx]
            )

        return self.graph
