# Training the node/edge classifier


## Annotated data prior to model training:

This README.md contains instructions on how to write a config file to train a classifier for image graph nodes and edges. To train the model based on image annotations, here's the list of data needed:

+ image data (in `mrc`, `tiff` or `png` formats), _e.g._ "Image_File_000.mrc"
+ corresponding `h5` file with the positions (xy+ coordinates)  of the detections on the image
+ corresponding `grace` folder with the same naming as the image, _e.g._ "Image_File_000.grace" which contains:
  + `annotation.npz` - masked object annotation array
  + `nodes.parquet` - nodes from the built graph, including their ground truth annotations
  + `edges.parquet` - edges from the built graph, including their ground truth annotations
  + `metadata.json` - information about the annotated file, such as `{"image_filename": "Image_File_000"}`


## Setting up the model configuration:

The following classifier is based on [graph convolutional network](https://arxiv.org/abs/1609.02907 "GCN seminal paper") architecture. To run the model training, you need to create a configuration (config) file where hyperparameters of the GCN model (or, potentially, other classifiers) and many other specs are specified:

The `config_file` should be structured as specified in the `config.json` or `config.yaml` file. See the [templates](./config.yaml) for detailed descriptions of the hyperparameters.


### Downloading the feature extractor:

In case you decide to use a pre-trained image classifier, such as resnet-152, you can use this snippet to import the model, load the default weights & download the model:

```sh
import torch
from grace.models.feature_extractor import resnet

resnet_model = resnet(resnet_type="resnet152")
extractor_fn = "/path/to/your/feature/extractor/resnet152.pt"
torch.save(resnet_model, extractor_fn)
```


## Full list of graph / patch augmentations:

| Augmentation  | Description                    | Parameters                   |
| ------------- | ------------------------------ | ---------------------------- |
| Graph         | `RandomXYTranslation`          | max_shift : float            |
| Graph         | `RandomEdgeAdditionAndRemoval` | p_add : float                |
|               |                                | p_remove : float             |
| Image         | `RandomEdgeCrop`               | max_fraction : float         |
| Image         | `RandomImageGraphRotate`       | rot_center : List[int]       |
|               |                                | rot_angle_range : Tuple[int] |


## Run the model training:

To run the model, you need to start the `run.py` script from the grace directory in the terminal. Prompt the script to locate the config file using the command line argument `config_file`:

```
python3 grace/training/run.py --config_file="/absolute/path/to/the/config/file/config.json"
```

### `Tensorboard` visualisation:

To launch `Tensorboard`, run this command in the command line:
```sh
tensorboard --logdir=/path/to/all/your/runs/
```

...and copy the `http://localhost:6006/` to your browser.

*Note:* If you desire to visualise multiple training run statistics, point the `logdir` to the directory one level above the run timestamp.
