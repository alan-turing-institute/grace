import torch

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

import numpy.typing as npt
import matplotlib

from sklearn.manifold import TSNE

from grace.models.datasets import dataset_from_graph
from grace.styling import LOGGER, COLORMAPS


def drop_linear_layers_from_model(
    model: torch.nn.Module,
) -> torch.nn.Sequential:
    """Chops off last 2 Linear layers from the classifier to
    access node embeddings learnt by the GCN classifier."""

    modules = list(model.children())[:-2]
    node_emb_extractor = torch.nn.Sequential(*modules)
    for p in node_emb_extractor.parameters():
        p.requires_grad = False

    return node_emb_extractor


class TSNEManifoldProjection(object):
    def __init__(
        self,
        graph: nx.Graph,
        model: str | Path,
    ) -> None:
        self.graph = graph
        self.model = model

        if self.model is not None:
            self.model = Path(self.model)
            assert self.model.is_file()

    def read_graph_dataset_IO(self) -> tuple[torch.stack]:
        # Prepare GT labels:
        dataset_batches = dataset_from_graph(
            graph=self.graph,
            mode="whole",
        )
        dataset_batches = dataset_batches[0]

        # Prepare the data:
        node_labels = dataset_batches.y
        node_embeds = dataset_batches.x
        edge_indices = dataset_batches.edge_index

        return node_labels, node_embeds, edge_indices

    def extract_GCN_node_embeddings(self) -> tuple[torch.stack]:
        node_labels, node_embeds, edge_indices = self.read_graph_dataset_IO()

        if self.model is None:
            LOGGER.info(
                "Warning, only returning the 'node_embeddings' as"
                "no pre-trained GCN model was specified..."
            )

        else:
            # Log the classifier time-stamp name:
            name = self.model.parent.name
            LOGGER.info(
                f"Processing the model time-stamp: '{name}/classifier.py'"
            )

            # Load the model & drop the `Linear` layers:
            full_gcn_classifier = torch.load(self.model)
            gcn_only_classifier = drop_linear_layers_from_model(
                full_gcn_classifier
            )

            # If only `Linear` model, log the warning:
            if len(gcn_only_classifier) < 1:
                LOGGER.info(
                    "Warning, only returning the 'node_embeddings' as "
                    "the GCN contains no graph convolutional layers..."
                )

            # Get the GCN node embeddings:
            else:
                # Prep the model & modify embeddings in-place:
                gcn_only_classifier.eval()
                for module in gcn_only_classifier[0]:
                    node_embeds = module(node_embeds, edge_indices)

            # Log the shapes:
            LOGGER.info(
                "Extracted 'node_embeddings' -> "
                f"{node_embeds.shape}, {node_embeds.dtype}"
            )

        return node_labels, node_embeds

    def perform_and_plot_tsne(
        self,
        node_GT_label: npt.NDArray,
        node_features: npt.NDArray,
        *,
        n_components: int = 2,
        title: str = "",
        cmap: str = COLORMAPS["manifold"],
        ax: matplotlib.axes = None,
    ) -> matplotlib.axes:
        # Shapes must agree:
        assert len(node_GT_label) == len(node_features)
        tsne = TSNE(n_components=n_components)
        node_embed = tsne.fit_transform(X=node_features)

        # Plot the TSNE manifold:
        title = f"TSNE of Patch Features\n{title}"
        umap1, umap2 = node_embed[:, 0], node_embed[:, 1]
        scatter = ax.scatter(x=umap1, y=umap2, c=node_GT_label, cmap=cmap)
        cbar = plt.colorbar(scatter)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel("Ground Truth Node Label", rotation=270)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title(title)
        return ax

    def plot_TSNE_before_and_after_GCN(self, **kwargs) -> None:
        # Plot the subplots:
        size = 5
        fig, axes = plt.subplots(1, 2, figsize=(size * 2 + 2, size * 1))

        # Get the embeddings:
        for p, (plot_name, method) in enumerate(
            zip(
                ["Before", "After"],
                [self.read_graph_dataset_IO, self.extract_GCN_node_embeddings],
            )
        ):
            labels, embeds = method()[:2]
            shape = embeds.shape[-1]
            title = f"{plot_name} GCN | Node Feature Embedding [{shape}]"
            self.perform_and_plot_tsne(
                labels, embeds, title=title, ax=axes[p], **kwargs
            )

        # Annotate & display:
        plt.tight_layout()
        return fig
