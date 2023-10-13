from pathlib import Path
from tqdm.auto import tqdm

from grace.styling import LOGGER
from grace.base import GraphAttrs, EdgeProps
from grace.models.datasets import dataset_from_graph

from grace.io.image_dataset import ImageGraphDataset
from grace.io.store_node_features import store_node_features_in_graph
from grace.io.store_edge_properties import store_edge_properties_in_graph


def check_and_chop_dataset(
    image_dir: str | Path,
    grace_dir: str | Path,
    filetype: str,
    node_feature_ndim: int,
    edge_property_len: int,
    keep_node_unknown_labels: bool,
    keep_edge_unknown_labels: bool,
    num_hops: int | str,
    connection: str = "spiderweb",
    store_permanently: bool = False,
    extractor_fn: str | Path = None,
):
    # Check if datasets are ready for training:
    dataset_ready_for_training = check_dataset_requirements(
        image_dir=image_dir,
        grace_dir=grace_dir,
        filetype=filetype,
        node_feature_ndim=node_feature_ndim,
        edge_property_len=edge_property_len,
    )
    if not dataset_ready_for_training:
        if store_permanently is True:
            assert extractor_fn is not None, "Provide feature extractor"

            # Inform the user about the delay:
            LOGGER.warning(
                "\n\nComputing node features & edge properties for data in "
                f"{grace_dir}. Expect to take ~30-40 seconds per file...\n\n"
            )
            store_node_features_in_graph(grace_dir, extractor_fn)
            store_edge_properties_in_graph(grace_dir)
        else:
            raise GraceGraphError(grace_dir=grace_dir)

    # Now that you have the files with node features & edge properties:
    target_list, subgraph_dataset = prepare_dataset_subgraphs(
        image_dir=image_dir,
        grace_dir=grace_dir,
        image_filetype=filetype,
        keep_node_unknown_labels=keep_node_unknown_labels,
        keep_edge_unknown_labels=keep_edge_unknown_labels,
        num_hops=num_hops,
        connection=connection,
    )
    return target_list, subgraph_dataset


def check_dataset_requirements(
    image_dir: str | Path,
    grace_dir: str | Path,
    filetype: str,
    node_feature_ndim: int,
    edge_property_len: int,
) -> tuple[list]:
    # Read the data & terate through images & extract node features:
    dataset_ready_for_training = True

    input_data = ImageGraphDataset(
        image_dir=image_dir,
        grace_dir=grace_dir,
        image_filetype=filetype,
        verbose=False,
    )

    # Process the (sub)graph data into torch_geometric dataset:
    desc = f"Chopping subgraphs from {grace_dir}"
    for _, target in tqdm(input_data, desc=desc):
        # Graph sanity checks: NODE_FEATURES:

        for _, node in target["graph"].nodes(data=True):
            if GraphAttrs.NODE_FEATURES not in node:
                dataset_ready_for_training = False
                break
            node_features = node[GraphAttrs.NODE_FEATURES]
            if node_features is None:
                dataset_ready_for_training = False
                break
            if node_features.shape[0] != node_feature_ndim:
                dataset_ready_for_training = False
                break

        # Graph sanity checks: EDGE_PROPERTIES:
        for _, _, edge in target["graph"].edges(data=True):
            if GraphAttrs.EDGE_PROPERTIES not in edge:
                dataset_ready_for_training = False
                break
            edge_properties = edge[GraphAttrs.EDGE_PROPERTIES]
            if edge_properties is None:
                dataset_ready_for_training = False
                break
            edge_properties = edge_properties.properties_dict
            if edge_properties is None:
                dataset_ready_for_training = False
                break
            if len(edge_properties) < edge_property_len:
                dataset_ready_for_training = False
                break
            if not all([item in edge_properties for item in EdgeProps]):
                dataset_ready_for_training = False
                break

    return dataset_ready_for_training


def prepare_dataset_subgraphs(
    image_dir: str | Path,
    grace_dir: str | Path,
    *,
    image_filetype: str,
    keep_node_unknown_labels: bool,
    keep_edge_unknown_labels: bool,
    num_hops: int | str,
    connection: str = "spiderweb",
) -> tuple[list]:
    # Read the data & terate through images & extract node features:
    input_data = ImageGraphDataset(
        image_dir=image_dir,
        grace_dir=grace_dir,
        image_filetype=image_filetype,
        keep_node_unknown_labels=keep_node_unknown_labels,
        keep_edge_unknown_labels=keep_edge_unknown_labels,
    )

    # Process the (sub)graph data into torch_geometric dataset:
    target_list, subgraph_dataset = [], []
    for _, target in input_data:
        # Store the valid graph list with the updated target:
        target_list.append(target)

        # Now, process the graph with all attributes & chop into subgraphs & store:
        graph_data = dataset_from_graph(
            target["graph"],
            num_hops=num_hops,
            connection=connection,
        )
        subgraph_dataset.extend(graph_data)

    return target_list, subgraph_dataset


class GraceGraphError(Exception):
    def __init__(self, grace_dir):
        super().__init__(
            "\n\nThe GRACE annotation files don't contain the proper node "
            "features & edge attributes for training \nin the `grace_dir` "
            f"= '{grace_dir}'\n\nPlease consider:\n\n(i) changing your config"
            " 'store_graph_attributes_permanently' argument to 'True', which "
            "will automatically compute & store the graph attributes & or \n"
            "(ii) manually run the scripts below for all your data paths, "
            "incl. 'train', 'valid' & 'infer' before launching the next run:"
            "\n\n\t`python3 grace/io/store_edge_properties.py --data_path="
            "/path/to/your/data` \nand"
            "\n\t`python3 grace/io/store_node_features.py --data_path="
            "/path/to/your/data --extractor_fn=/path/to/feature/extractor.pt`"
            "\n\nThis will compute required graph attributes & store them "
            "in the GRACE annotation file collection, avoiding this error.\n"
        )
