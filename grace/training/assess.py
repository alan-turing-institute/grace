from pathlib import Path

from grace.styling import LOGGER
from grace.models.optimiser import optimise_graph

from grace.training.config import write_file_with_suffix
from grace.evaluation.inference import GraphLabelPredictor
from grace.evaluation.process import generate_ground_truth_graph
from grace.evaluation.metrics_objects import ExactMetricsComputer


def assess_training_performance(
    run_dir: str | Path,
    infer_target_list: list[dict],
    compute_exact_metrics: bool,
    compute_approx_metrics: bool,
) -> None:
    """Callable to perform GRACE inference with a trained classifier model.

    Parameters
    ----------
    run_dir : str | Path
        Pointer to the time-stamped directory of training run. Structure:
        |-- time-stamp (e.g. 2023-10-25_18-00-00)
            |-- events.out.tfevents...
            |-- model
                |-- classifier.pt
                |-- config_hyperparams.json
                |-- config_hyperparams.yaml
                |-- summary_architecture.txt
            |-- infer
            |-- valid
            |-- weights (optional)
    """
    # Run inference on the final, trained model on unseen data:
    GLP = GraphLabelPredictor(run_dir / "model" / "classifier.pt")

    # Process entire batch & save the results:
    inference_metrics = GLP.calculate_numerical_results_on_entire_batch(
        infer_target_list,
    )
    # Log inference metrics:
    LOGGER.info(f"Inference dataset batch metrics: {inference_metrics}")

    # Write out the batch metrics:
    batch_metrics_fn = run_dir / "infer" / "Batch_Dataset-Metrics.json"
    write_file_with_suffix(inference_metrics, batch_metrics_fn)
    batch_metrics_fn = run_dir / "infer" / "Batch_Dataset-Metrics.yaml"
    write_file_with_suffix(inference_metrics, batch_metrics_fn)

    # Save out the inference batch performance figures:
    GLP.visualise_model_performance_on_entire_batch(
        infer_target_list, save_figures=run_dir / "infer", show_figures=False
    )

    # Process each inference batch file individually:
    for i, graph_data in enumerate(infer_target_list):
        progress = f"[{i+1} / {len(infer_target_list)}]"
        fn = graph_data["metadata"]["image_filename"]
        LOGGER.info(f"{progress} Processing file: '{fn}'")

        infer_graph = graph_data["graph"]
        GLP.set_node_and_edge_probabilities(G=infer_graph)
        GLP.visualise_prediction_probs_on_graph(
            G=infer_graph,
            graph_filename=fn,
            save_figure=run_dir / "infer",
            show_figure=False,
        )

        # Try visualising the attention weights (skips if None):
        GLP.visualise_attention_weights_on_graph(
            G=infer_graph,
            graph_filename=fn,
            save_figure=run_dir / "infer",
            show_figure=False,
        )

        # Generate GT & optimised graphs:
        true_graph = generate_ground_truth_graph(infer_graph)
        pred_graph = optimise_graph(infer_graph)

        # EXACT metrics per image:
        if compute_exact_metrics is True:
            EMC = ExactMetricsComputer(
                G=infer_graph,
                pred_optimised_graph=pred_graph,
                true_annotated_graph=true_graph,
            )

            # Compute EXACT numerical metrics & write them out as file:
            EMC_metrics = EMC.metrics()
            LOGGER.info(f"{progress} Exact metrics: {fn} | {EMC_metrics}")

            EMC_fn = run_dir / "infer" / f"{fn}-Metrics.json"
            write_file_with_suffix(EMC_metrics, EMC_fn)
            EMC_fn = run_dir / "infer" / f"{fn}-Metrics.yaml"
            write_file_with_suffix(EMC_metrics, EMC_fn)

            EMC.visualise(
                save_path=run_dir / "infer",
                file_name=fn,
                save_figures=True,
                show_figures=False,
            )

        # APPROX metrics per image:
        if compute_approx_metrics is True:
            LOGGER.warning(
                f"{progress} WARNING; 'APPROX' metrics not implemented yet"
            )
