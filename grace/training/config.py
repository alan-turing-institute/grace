from typing import Any, Union, Optional

import os
import json
import yaml

from pathlib import Path
from dataclasses import field, dataclass
from grace.styling import LOGGER


@dataclass
class Config:
    train_image_dir: Optional[os.PathLike] = None
    train_grace_dir: Optional[os.PathLike] = None
    valid_image_dir: Optional[os.PathLike] = None
    valid_grace_dir: Optional[os.PathLike] = None
    infer_image_dir: Optional[os.PathLike] = None
    infer_grace_dir: Optional[os.PathLike] = None
    extractor_fn: Optional[os.PathLike] = None
    log_dir: Optional[os.PathLike] = None
    run_dir: Optional[os.PathLike] = log_dir
    filetype: str = "mrc"
    normalize: tuple[bool] = (False, False)
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
    patch_size: tuple[int] = (224, 224)
    keep_patch_fraction: float = 1.0
    keep_node_unknown_labels: bool = False
    keep_edge_unknown_labels: bool = False
    feature_dim: int = 2048

    classifier_type: str = "GCN"
    num_node_classes: int = 2
    num_edge_classes: int = 2
    epochs: int = 100
    hidden_channels: list[int] = field(default_factory=lambda: [1024, 256, 64])
    metrics_classifier: list[str] = field(
        default_factory=lambda: ["accuracy", "f1_score", "confusion_matrix"]
    )
    metrics_objects: list[str] = field(
        default_factory=lambda: ["exact", "approx"]
    )
    dropout: float = 0.2
    batch_size: int = 64
    learning_rate: float = 0.001
    tensorboard_update_frequency: int = 1
    valid_graph_ploter_frequency: int = 1
    animate_valid_progress: bool = False
    visualise_tsne_manifold: bool = False
    # saving_file_suffix: str = "yaml"


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
    if not config.extractor_fn.is_file():
        raise PathNotDefinedError(path_name=dr)

    # Define which metrics to calculate:
    for i in range(len(config.metrics_objects)):
        m = config.metrics_objects[i].upper()
        if m == "APPROXIMATE":
            m = "APPROX"
        config.metrics_objects[i] = m

    # Make sure saving file suffix is expected:
    # assert config.saving_file_suffix in {"yaml", "json"}

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


def write_config_file(
    config: Config,
    filetype: str = "json",
) -> None:
    """Record hyperparameters of a training run."""
    params = {attr: str(getattr(config, attr)) for attr in config.__dict__}

    if isinstance(config.run_dir, str):
        setattr(config, "run_dir", Path(config.run_dir))

    fn = config.run_dir / f"config_hyperparams.{filetype}"
    write_params_as_file_with_suffix(params, fn)


def write_params_as_file_with_suffix(
    parameters_dict: dict[str], filename: str | Path
) -> None:
    if isinstance(filename, str):
        filename = Path(filename)

    if filename.suffix == ".json":
        with open(filename, "w") as outfile:
            json.dump(parameters_dict, outfile, indent=4)

    elif filename.suffix == ".yaml":
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
