import numpy as np
import numpy.typing as npt
import networkx as nx
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.ndimage import label

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

from grace.visualisation.plotting import plot_optimised_objects_from_graphs
from grace.visualisation.utils import (
    intersection_over_union,
    list_real_connected_components,
    find_matching_pairs,
    format_object_detection_metrics,
    plot_iou_histogram,
    plot_confusion_matrix_tiles,
    visualise_semantic_iou_map,
    visualise_bounding_boxes_on_graph,
)


class ExactMetricsComputer(object):
    """Metrics wrapper to calculate & visualise GRACE pipeline performance.
        EXACT metrics compare two graphs with identical nodes positions
        which differ only by presence / absence of edges, defining objects.

    Parameters
    ----------
    G : nx.Graph
        Original graph with all nodes and possible edges (as triangulated)
    pred_optimised_graph : nx.Graph
        Predicted graph with individual objects after the optimisation step.
    true_annotated_graph : nx.Graph
        Ground truth graph with individual objects from napari annotation.

    Methods
    -------
    metrics(print_results: bool = False)
        - Computes numerical metrics, returned as dict[str, float]
          For nodes: accuracy, precision, recall, f1-score
          For edges: accuracy, precision, recall, f1-score
          IoU semantic, instance [mean ± st.dev], list of object IoU scores
        - Optionally prints metrics in a formatted table. Defaults to False.

    visualise()
        - Visualises per-node & per-edge raw & normalised confusion matrix.
        - Visualises IoU metrics histogram per individual object score.
        - Visualises IoU object overlap with GT as bbox on top of the graph.
    """

    def __init__(
        self,
        G: nx.Graph,
        pred_optimised_graph: nx.Graph,
        true_annotated_graph: nx.Graph,
    ) -> None:
        # Instantiate metrics attributes:
        self.graph = G
        self.pred_graph = pred_optimised_graph
        self.true_graph = true_annotated_graph

    @staticmethod
    def _pool_connected_components_nodes(G: nx.Graph) -> set[int]:
        con_pool = set()
        cc = nx.connected_components(G)
        for comp in cc:
            # Ignore 1-node components:
            if len(comp) > 1:
                con_pool.update(comp)
        return con_pool

    @staticmethod
    def _pool_connected_components_edges(
        G: nx.Graph,
    ) -> set[list[tuple[int, int], tuple[int, int]]]:
        con_pool = set()
        cc = nx.connected_components(G)
        for comp in cc:
            if len(comp) > 1:
                for i in range(len(list(comp)) - 1):
                    src, dst = list(comp)[i : i + 2]
                    con_pool.update([(src, dst), (dst, src)])  # bi-directional
                    # con_pool.update((src, dst))
        return con_pool

    @staticmethod
    def _generate_label_vector(input_data_set: set, component_pool_set: set):
        y_labels = [n in component_pool_set for n in input_data_set]
        y_labels = np.array(y_labels).astype(int)
        return y_labels

    def process_graph_elements_data(self, element_type: str):
        if element_type == "nodes":
            all_triangulated_element = self.graph.nodes(data=False)
            con_comp_function = self._pool_connected_components_nodes

        elif element_type == "edges":
            all_triangulated_element = self.graph.edges(data=False)
            con_comp_function = self._pool_connected_components_edges

        else:
            raise ValueError(f"Unspecified graph element: '{element_type}'")

        # Process the data:
        con_pool_pred = con_comp_function(self.pred_graph)
        con_pool_true = con_comp_function(self.true_graph)

        y_pred = self._generate_label_vector(
            input_data_set=all_triangulated_element,
            component_pool_set=con_pool_pred,
        )
        y_true = self._generate_label_vector(
            input_data_set=all_triangulated_element,
            component_pool_set=con_pool_true,
        )
        return y_pred, y_true

    def calculate_numerical_results(self, element_type: str):
        y_pred, y_true = self.process_graph_elements_data(
            element_type=element_type
        )
        # Accuracy, precision, recall, f1, support scores:
        accuracy = accuracy_score(y_pred=y_pred, y_true=y_true, normalize=True)
        prec_recall_f1score = precision_recall_fscore_support(
            y_pred=y_pred,
            y_true=y_true,
            average="weighted",
            beta=1.0,
            zero_division=0.0,
        )
        return accuracy, prec_recall_f1score

    def calculate_iou_results(self) -> tuple[float, list[float]]:
        # Semantic IoU:
        cc_set_pred = self._pool_connected_components_nodes(self.pred_graph)
        cc_set_true = self._pool_connected_components_nodes(self.true_graph)
        iou_semantic = intersection_over_union(cc_set_pred, cc_set_true)

        # Instance IoU:
        cc_list_pred = list_real_connected_components(self.pred_graph)
        cc_list_true = list_real_connected_components(self.true_graph)

        # Find matching pairs of sets with at least one common node
        matching_pairs = find_matching_pairs(cc_list_pred, cc_list_true)

        # Calculate IoU for every component pair with at least 1-node overlap:
        iou_per_object = []
        for pair in matching_pairs:
            iou_score = intersection_over_union(pair[0], pair[1])
            iou_per_object.append(iou_score)

        return iou_semantic, iou_per_object

    def plot_confusion_matrix(
        self,
        figsize: tuple[int, int] = (10, 10),
        colormap: str = "copper",
    ) -> None:
        # Prepare the data:
        node_pred, node_true = self.process_graph_elements_data("nodes")
        edge_pred, edge_true = self.process_graph_elements_data("edges")
        plot_confusion_matrix_tiles(
            node_pred,
            edge_pred,
            node_true,
            edge_true,
            figsize,
            colormap,
        )

    def metrics(self, print_results: bool = False) -> dict[str, float]:
        results_dict = {}

        # Object detection (nodes & edges)
        for element_type in ["nodes", "edges"]:
            acc, prf1 = self.calculate_numerical_results(element_type)

            title = f"Object Detection Metrics | {element_type.capitalize()}"
            table = {
                f"Accuracy ({element_type})": acc,
                f"Precision ({element_type})": prf1[0],
                f"Recall ({element_type})": prf1[1],
                f"F1-Score ({element_type})": prf1[2],
            }
            # Format & print the table:
            if print_results is True:
                formatted_table = format_object_detection_metrics(table, title)
                print(formatted_table)

            # Append the whole instance IoU list to the dict:
            results_dict.update(table)

        # Intersection over union:
        iou_semantic, iou_per_object = self.calculate_iou_results()

        title = "Intersection over Union"
        table = {
            "Semantic IoU": iou_semantic,
            "Instance IoU [mean]": np.mean(iou_per_object),
            "Instance IoU [std]": np.std(iou_per_object),
        }
        # Format & print the table:
        if print_results is True:
            formatted_table = format_object_detection_metrics(table, title)
            print(formatted_table)

        # Append the whole instance IoU list to the dict:
        results_dict.update(table)
        results_dict["Instance IoU [list]"] = iou_per_object

        # Turn all values into floats where appropriate:
        for key, value in results_dict.items():
            if isinstance(value, float):
                results_dict[key] = float(value)
            elif isinstance(value, list):
                results_dict[key] = [float(v) for v in value]

        return results_dict

    def visualise(
        self,
        save_path: str = None,
        file_name: str = None,
        save_figures: bool = False,
        show_figures: bool = False,
    ) -> None:
        # Calculate the metrics:
        metric_dict = self.metrics()

        if save_figures is True:
            if isinstance(save_path, str):
                save_path = Path(save_path)
                assert save_path.is_dir()

        # Confusion matrices:
        # self.plot_confusion_matrix()

        # IoU scores histogram:
        plot_iou_histogram(
            iou_per_object=metric_dict["Instance IoU [list]"],
            iou_semantic=metric_dict["Semantic IoU"],
        )
        if save_figures is True:
            plt.savefig(save_path / f"{file_name}-Object_IoU_Histogram.png")
        if show_figures is True:
            plt.show()
        plt.close()

        # Plot bbox overlap on mask:
        visualise_bounding_boxes_on_graph(
            self.graph, self.pred_graph, self.true_graph, display_image=None
        )
        if save_figures is True:
            plt.savefig(save_path / f"{file_name}-Object_Bounding_Boxes.png")
        if show_figures is True:
            plt.show()
        plt.close()

        # Save out the connected components figure:
        plot_optimised_objects_from_graphs(
            triangulated_graph=self.graph,
            true_graph=self.true_graph,
            pred_graph=self.pred_graph,
        )
        if save_figures is True:
            plt.savefig(save_path / f"{file_name}-Optimised_Components.png")
        if show_figures is True:
            plt.show()
        plt.close()


class ApproxMetricsComputer(object):
    """Metrics wrapper to calculate & visualise GRACE pipeline performance.
        APPROX metrics compare two annotation masks & ovelapping objects.

    Parameters
    ----------
    G : nx.Graph
        Original graph with all nodes and possible edges (as triangulated)
    pred_auto_annotated_mask : npt.NDArray
        Predicted, automatically drown annotation mask from optimised graph;
        check out `draw_annotation_mask_from_ground_truth_graph` function
    true_hand_annotated_mask : npt.NDArray
        Ground truth annotation mask with annotated objects, e.g. from napari

    Methods
    -------
    metrics(print_results: bool = False)
        - Computes numerical metrics, returned as dict[str, float]
          IoU semantic, instance [mean ± st.dev], list of object IoU scores
        - Optionally prints metrics in a formatted table. Defaults to False.

    visualise()
        - Visualises IoU metrics histogram per individual object score.
        - Visualises colour-coded annotation overlap for visual inspection.
        - Visualises IoU object overlap with GT as bbox on top of the graph.
    """

    def __init__(
        self,
        G: nx.Graph,
        pred_auto_annotated_mask: npt.NDArray,
        true_hand_annotated_mask: npt.NDArray,
    ) -> None:
        # Instantiate metrics attributes:
        self.graph = G
        self.auto_anno = pred_auto_annotated_mask
        self.hand_anno = true_hand_annotated_mask

    def _calculate_pixel_accuracy(self):
        agreement_mask = self.hand_anno == self.auto_anno
        return np.sum(agreement_mask) / agreement_mask.size

    def _calculate_object_accuracy(self):
        overlap_mask = np.logical_and(self.hand_anno, self.auto_anno)
        agreement_mask = np.logical_or(self.hand_anno, self.auto_anno)
        return np.sum(overlap_mask) / np.sum(agreement_mask)

    def _create_overlay_image(self):
        # Create channels for red, green, and blue
        r_channel = np.where(~self.hand_anno & self.auto_anno, 255, 0)
        g_channel = np.where(self.hand_anno & ~self.auto_anno, 255, 0)
        b_channel = np.where(self.hand_anno & self.auto_anno, 255, 0)

        # Combine the channels to form an RGB image
        overlay_image = np.stack((r_channel, g_channel, b_channel), axis=-1)
        return overlay_image

    def semantic_iou_from_masks(self) -> tuple[float]:
        # Calculate accuracy metrics
        acc_pix = self._calculate_pixel_accuracy()
        acc_obj = self._calculate_object_accuracy()
        return acc_pix, acc_obj

    def instance_iou_from_masks(self) -> npt.NDArray:
        mask_sum = np.add(self.hand_anno, self.auto_anno)
        labeled_map, num_objects = label(mask_sum)

        iou_list = []
        for lab_idx in range(1, num_objects + 1):
            obj_mask = labeled_map == lab_idx
            union_lab = np.sum(obj_mask)

            agreed = self.hand_anno[obj_mask] == self.auto_anno[obj_mask]
            inter_lab = np.sum(agreed)

            iou_score = inter_lab / union_lab
            iou_list.append(iou_score)

        return iou_list

    def metrics(self, print_results: bool = False) -> dict[str, float]:
        # Calculate IoUs:
        pixel_accuracy, object_accuracy = self.semantic_iou_from_masks()
        iou_per_object = self.instance_iou_from_masks()

        # Organise into dictionary & print numerical values:
        title = "Intersection over Union (Approx)"
        table = {
            "Pixel Accuracy": pixel_accuracy,
            "Semantic IoU": object_accuracy,
            "Instance IoU [mean]": np.mean(iou_per_object),
            "Instance IoU [std]": np.std(iou_per_object),
        }
        # Format & print the table:
        if print_results is True:
            formatted_table = format_object_detection_metrics(table, title)
            print(formatted_table)

        # Append the whole instance IoU list to the dict:
        table["Instance IoU [list]"] = iou_per_object
        return table

    def visualise(self) -> None:
        # Calculate the metrics:
        metric_dict = self.metrics()

        # Plot numerical values:
        plot_iou_histogram(
            iou_per_object=metric_dict["Instance IoU [list]"],
            iou_semantic=metric_dict["Semantic IoU"],
        )

        # Visualise semantic IoU:
        visualise_semantic_iou_map(self.auto_anno, self.hand_anno)

        # Plot bbox overlap on mask:
        visualise_bounding_boxes_on_graph(
            self.graph,
            self.auto_anno,
            self.hand_anno,
            display_image=self.hand_anno,
        )
