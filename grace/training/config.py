from typing import Any, Union, Optional

import os
import json
import yaml

from pathlib import Path
from dataclasses import field, dataclass
from grace.styling import LOGGER


@dataclass
class Config:
    # Paths to inputs & outputs:
    train_image_dir: Optional[os.PathLike] = None
    train_grace_dir: Optional[os.PathLike] = None
    valid_image_dir: Optional[os.PathLike] = None
    valid_grace_dir: Optional[os.PathLike] = None
    infer_image_dir: Optional[os.PathLike] = None
    infer_grace_dir: Optional[os.PathLike] = None
    log_dir: Optional[os.PathLike] = None
    run_dir: Optional[os.PathLike] = log_dir

    # Feature extraction:
    filetype: str = "mrc"
    keep_node_unknown_labels: bool = False
    keep_edge_unknown_labels: bool = False

    # Feature extraction:
    extractor_fn: Optional[os.PathLike] = None
    patch_size: tuple[int] = (224, 224)
    feature_dim: int = 2048
    normalize: tuple[bool] = (False, False)

    # Augmentations:
    img_graph_augs: list[str] = field(
        default_factory=lambda: [
            "random_edge_addition_and_removal",
            "random_xy_translation",
            "random_image_graph_rotate",
        ]
    )
    img_graph_aug_params: list[dict[str, Any]] = field(
        default_factory=lambda: [{}, {}, {}]
    )
    patch_augs: list[str] = field(
        default_factory=lambda: [
            "random_edge_crop",
        ]
    )
    patch_aug_params: list[dict[str, Any]] = field(
        default_factory=lambda: [{}]
    )
    keep_patch_fraction: float = 1.0

    # Classifier architecture setup
    classifier_type: str = "GCN"
    num_node_classes: int = 2
    num_edge_classes: int = 2
    hidden_graph_channels: list[int] = field(
        default_factory=lambda: [1024, 256, 64]
    )
    hidden_dense_channels: list[int] = field(
        default_factory=lambda: [1024, 256, 64]
    )

    # Training run hyperparameters:
    batch_size: int = 64
    epochs: int = 100
    dropout: float = 0.2
    learning_rate: float = 0.001
    weight_decay: float = 0.0

    # Learning rate scheduler:
    scheduler_type: str = "none"
    scheduler_step: int = 1
    scheduler_gamma: float = 1.0

    # Performance evaluation:
    metrics_classifier: list[str] = field(
        default_factory=lambda: ["accuracy", "f1_score", "confusion_matrix"]
    )
    metrics_objects: list[str] = field(
        default_factory=lambda: ["exact", "approx"]
    )

    # Validation & visualisation:
    tensorboard_update_frequency: int = 1
    valid_graph_ploter_frequency: int = 1
    animate_valid_progress: bool = False
    visualise_tsne_manifold: bool = False


def load_config_params(params_file: Union[str, Path]) -> Config:
    """Overwrite default config params from a
    JSON or YAML file.

    Parameters
    ----------
    params_file : Union[str, Path]
        File specifying params to update; or directory containing
        this file

    Returns
    -------
    config : Config
        Updated config
    """

    config = Config()

    if isinstance(params_file, str):
        params_file = Path(params_file)

    if not params_file.is_file():
        try:
            params_file = list(Path(params_file).glob("config_hyperparams.*"))[
                0
            ]
        except IndexError:
            raise ValueError("Config file cannot be found in this directory.")

    if params_file.suffix == ".json":
        params_dict = json.load(open(params_file))

    elif params_file.suffix == ".yaml":
        params_dict = yaml.safe_load(open(params_file))

    else:
        raise ValueError("Params file must be either a .json or .yaml file.")

    for attr in config.__dict__:
        if attr in params_dict:
            default_value = getattr(config, attr)
            value = params_dict[attr]

            if attr.endswith("_dir") or attr.endswith("_fn"):
                value = Path(value)

            if params_file.suffix == ".json":
                if default_value is None:
                    value = value
                elif isinstance(default_value, (bool, list, tuple)):
                    value = eval(value)
                else:
                    value = type(default_value)(value)

            setattr(config, attr, value)

    return config


def validate_required_config_hparams(config: Config) -> None:
    # Check all required directories are defined:
    directories = [
        config.train_image_dir,
        config.train_grace_dir,
        config.valid_image_dir,
        config.valid_grace_dir,
        config.infer_image_dir,
        config.infer_grace_dir,
    ]
    for dr in directories:
        if dr is None:
            raise PathNotDefinedError(path_name=dr)
        elif not any(dr.iterdir()):
            raise EmptyDirectoryError(path_name=dr)
        else:
            pass

    # Check log_dir exists:
    if config.log_dir is None:
        raise PathNotDefinedError(path_name=dr)

    # Check extractor is there:
    if config.extractor_fn is not None:
        if not config.extractor_fn.is_file():
            raise PathNotDefinedError(path_name=dr)

    # Check that hidden_channels are all integers:
    assert all(isinstance(ch, int) for ch in config.hidden_graph_channels)
    assert all(isinstance(ch, int) for ch in config.hidden_dense_channels)

    # Validate the learning rate schedule is implemented:
    assert config.scheduler_type in {"none", "step", "expo"}

    # Define which object metrics to calculate:
    for i in range(len(config.metrics_classifier)):
        m = config.metrics_classifier[i].lower()
        config.metrics_classifier[i] = m
    assert all(
        m in {"accuracy", "f1_score", "confusion_matrix"}
        for m in config.metrics_classifier
    )

    # Define which object metrics to calculate:
    for i in range(len(config.metrics_objects)):
        m = config.metrics_objects[i].lower()
        if m == "approximate":
            m = "approx"
        config.metrics_objects[i] = m
    assert all(m in {"exact", "approx"} for m in config.metrics_objects)

    # HACK: not automated yet:
    if config.animate_valid_progress is True:
        LOGGER.warning("WARNING; auto-animation not implemented yet")
        config.animate_valid_progress = False
        # TODO: implemented, but ffmpeg causes issues in tests

    # HACK: not implemented yet:
    if config.visualise_tsne_manifold is True:
        LOGGER.warning("WARNING; TSNE manifold not implemented yet")
        config.visualise_tsne_manifold = False
        # TODO: implemented, but can't be run from run.py yet


def write_config_file(config: Config, filetype: str = "json") -> None:
    """Record hyperparameters of a training run."""
    params = {attr: getattr(config, attr) for attr in config.__dict__}

    if isinstance(config.run_dir, str):
        setattr(config, "run_dir", Path(config.run_dir))

    fn = config.run_dir / f"config_hyperparams.{filetype}"
    write_params_as_file_with_suffix(params, fn)


def write_params_as_file_with_suffix(
    parameters_dict: dict[Any], filename: str | Path
) -> None:
    if isinstance(filename, str):
        filename = Path(filename)

    if filename.suffix == ".json":
        # Convert all params to strings:
        for attr, param in parameters_dict.items():
            parameters_dict[attr] = str(param)
        # Write the file out:
        with open(filename, "w") as outfile:
            json.dump(
                parameters_dict,
                outfile,
                indent=4,
            )

    elif filename.suffix == ".yaml":
        # Convert all params to yaml-parsable types:
        for attr, param in parameters_dict.items():
            if isinstance(param, Path):
                parameters_dict[attr] = str(param)
            elif isinstance(param, tuple):
                parameters_dict[attr] = list(param)
            else:
                parameters_dict[attr] = param
        # Write the file out in human-readable form:
        with open(filename, "w") as outfile:
            yaml.dump(
                parameters_dict,
                outfile,
                default_flow_style=False,
                allow_unicode=True,
            )

    else:
        ValueError("Filetype suffix must be 'json' or 'yaml'.")


class PathNotDefinedError(Exception):
    def __init__(self, path_name):
        super().__init__(f"The path '{path_name}' is not defined.")


class EmptyDirectoryError(Exception):
    def __init__(self, path_name):
        super().__init__(f"The path '{path_name}' is empty.")
