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

The `config_file` should be structured as follows:

```
{
    # Paths to data:
    "image_dir": "/absolute/path/to/your/images/",
    "grace_dir": "--same--as--above--",
    "run_dir": "/absolute/path/to/where/runs/will/be/saved/",
    "log_dir": "--same--as--above--",

    # Feature extractor:
    "extractor_fn": "/path/to/your/feature/extractor/resnet152.pt",
    "feature_dim": "2048", -> if using ResNet 50 / 101 / 152,
                   "512" if using ResNet 18 / 34; = input channels into the classifier

    # Patch data specs:
    "patch_size": "(224, 224)", -> size of the patch to crop & feed to feature extractor
    "keep_patch_fraction": "1.0", -> required fraction of the image not to be excluded

    # Groups of augmentations [see [table](table) below]:
    # see 'grace.utils/augment_graph.py' for full option list
    "img_graph_augs": "['random_edge_addition_and_removal', 'random_xy_translation']",
    "img_graph_aug_params": "[{'p_add': 0.2}, {'max_shift': 15.0}]",

    # see 'grace.utils/augment_image.py' for full option list
    "patch_augs": "[]",
    "patch_aug_params": "[]",

    # Classifier model training:
    "epochs": "10",
    "num_node_classes": "2",
    "num_edge_classes": "2",
    "hidden_channels": "[512, 128, 32]",
    "metrics": "['accuracy', 'confusion_matrix']",
}
```

_Note:_ Write the parameters into a single line, the file will be parsed accordingly.

Downloading the feature extractor:

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
python3 grace/run.py --config_file="/absolute/path/to/the/config/file/config.json"
```
_e.g._
```
python3 grace/run.py --config_file=/Users/kulicna/Desktop/classifier/configs/config.json
```


# Candidates for hyperparameters:

+ learning rate
+ dropout probability
+ train to validation ratio
  + these should be updated from the config  
+ specify which model you want to run: "gcn", "gat", "linear classifier", "tsne", "central patch pixel"
+ clean the optimiser - make the inference function as `predict`
  + set the model to eval()
  + set the dropout to 0
