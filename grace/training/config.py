from typing import Any, List, Dict, Tuple, Union, Optional

import os
import json
import yaml

from pathlib import Path
from dataclasses import field, dataclass


@dataclass
class Config:
    image_dir: Optional[os.PathLike] = None
    grace_dir: Optional[os.PathLike] = None
    log_dir: Optional[os.PathLike] = None
    run_dir: Optional[os.PathLike] = None
    filetype: str = "mrc"
    img_graph_augs: List[str] = field(
        default_factory=lambda: [
            "random_edge_addition_and_removal",
            "random_xy_translation",
            "random_image_graph_rotate",
        ]
    )
    img_graph_aug_params: List[Dict[str, Any]] = field(
        default_factory=lambda: [{}, {}, {}]
    )
    extractor_fn: Optional[os.PathLike] = None
    patch_augs: List[str] = field(
        default_factory=lambda: [
            "random_edge_crop",
        ]
    )
    patch_aug_params: List[Dict[str, Any]] = field(
        default_factory=lambda: [{}]
    )
    patch_size: Tuple[int] = (224, 224)
    keep_patch_fraction: float = 1.0
    keep_unknown_labels: bool = False
    feature_dim: int = 2048
    num_node_classes: int = 2
    num_edge_classes: int = 2
    epochs: int = 100
    hidden_channels: List[int] = field(default_factory=lambda: [1024, 256, 64])
    metrics: List[str] = field(
        default_factory=lambda: ["accuracy", "confusion_matrix"]
    )
    dropout: float = 0.5
    batch_size: int = 64
    learning_rate: float = 0.001
    tensorboard_update_frequency: int = 1


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

            if attr.endswith("_dir"):
                value = Path(value)
            elif default_value is None:
                value = value
            elif isinstance(default_value, (bool, list, tuple)):
                value = eval(value)
            else:
                value = type(default_value)(value)

            setattr(config, attr, value)

        else:
            continue

    return config


def write_config_file(
    config: Config,
    filetype: str = "json",
) -> None:
    """Record hyperparameters of a training run."""
    if filetype not in ["json", "yaml"]:
        raise ValueError(
            "Config must be saved as either a .json or .yaml file."
        )

    """params = {}

    for attr in config.__dict__:

        value = getattr(config, attr)
        if isinstance(value, function):
            value = []

        params[attr] = str(value)"""

    params = {attr: str(getattr(config, attr)) for attr in config.__dict__}

    if isinstance(config.run_dir, str):
        setattr(config, "run_dir", Path(config.run_dir))

    fn = config.run_dir / f"config_hyperparams.{filetype}"

    with open(fn, "w") as outfile:
        if filetype == "json":
            json.dump(params, outfile, indent=4)
        else:
            yaml.dump(params, outfile)
