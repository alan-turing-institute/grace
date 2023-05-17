from typing import Union

import os
import click
import torch

from datetime import datetime


from grace.config import write_config_file, load_config_params
from grace.io.image_dataset import ImageGraphDataset
from grace.models.train import train_model
from grace.models.datasets import dataset_from_graph
from grace.models.classifier import GCN
from grace.models.feature_extractor import FeatureExtractor
from grace.utils.transforms import get_transforms

@click.command(name="GRACE Trainer")
@click.option("--config_file", type=click.Path(exists=True))
def run(config_file: Union[str, os.PathLike]) -> None:
    """Runs the GRACE pipeline; going straight from images and annotations
    to a trained node/edge classifier model.

    This function sacrifices some flexibility
    
    Parameters
    ----------
    config_file : str or Path
        Config file specifying the hyperparameters of the model, training run
        and dataset loading procedure. If this is a Path or string, then it is 
        assumed, this will be the filename of a config file (.json or .yaml).
    """
    config = load_config_params(config_file)
    
    extractor_model = torch.load(config.extractor_fn)
    patch_augs = get_transforms(config, "patch")
    img_graph_augs = get_transforms(config, "graph")
    feature_extractor = FeatureExtractor(model=extractor_model,
                                         augmentations=patch_augs,
                                         bbox_size=config.patch_size,
                                         ignore_fraction=config.ignore_fraction,)
    
    transform = lambda x: feature_extractor(img_graph_augs(x))
    input_data = ImageGraphDataset(image_dir=config.image_dir,
                                   grace_dir=config.grace_dir,
                                   filetype=config.filetype,
                                   transform=transform,)
    
    dataset = []

    for _, target in input_data:

        dataset.extend(
            dataset_from_graph(target["graph"])
        )

    classifier = GCN(input_channels=config.feature_dim,
                     hidden_channels=config.hidden_channels,
                     node_output_classes=config.num_node_classes,
                     edge_output_classes=config.num_edge_classes,)
    
    current_time = datetime.now().strftime("%d-%b-%Y_%H-%M-%S")
    run_dir = config.log_dir / current_time
    setattr(config, "run_dir", run_dir)
    
    train_model(classifier, 
                dataset, 
                epochs=config.epochs, 
                log_dir=run_dir, 
                metrics=config.metrics)
    
    model_save_fn = run_dir / "classifier.pt"
    torch.save(classifier, model_save_fn)
    write_config_file(config)
    
if __name__ == "__main__":
    run()