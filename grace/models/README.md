# Training the node/edge classifier


## Annotated data prior to model training:

The following classifier is based on [graph convolutional network](https://arxiv.org/abs/1609.02907 "GCN seminal paper") architecture. To train the model based on image annotations, here's the necessary data needed:

+ image data (in `mrc`, `tiff` or `png` formats), _e.g._ "Image_File_000.mrc"
+ corresponding `h5` file with the positions (xy+ coordinates)  of the detections on the image
+ corresponding `grace` folder with the same naming as the image, _e.g._ "Image_File_000.grace" which contains:
  + `annotation.npz` - masked object annotation array
  + `nodes.parquet` - nodes from the built graph, including their ground truth annotations
  + `edges.parquet` - edges from the built graph, including their ground truth annotations
  + `metadata.json` - information about the annotated file, such as `{"image_filename": "Image_File_000"}`


## Setting up the model configuration:

To run the model training, you need to create a configuration (config) file where hyperparameters of the model and many other specs are specified:


absolute path to the `config_file` structured as follows:

```
{
    # Paths to data:
    "image_dir": "/absolute/path/to/your/images/",
    "grace_dir": "--same--as--above--",
    "run_dir": "/absolute/path/to/where/runs/will/be/saved/", "log_dir": "--same--as--above--",

    # Feature extractor:
    "extractor_fn": "None",
    "feature_dim": "2048",

    # Patch data specs:
    "patch_size": "(224, 224)",
    "keep_patch_fraction": "1.0", -> required fraction of the image not to be excluded

    # Groups of augmentations:
    "img_graph_augs": "['random_edge_addition_and_removal', 'random_xy_translation']",
    "img_graph_aug_params": "[{'p_add': 0.2}, {'max_shift': 15.0}]",

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

Training takes a while to start, perhaps consider including the progress logging?


## Running the model:

To run the model, you need to start the `run.py` script from the grace directory in the terminal. Prompt the script to locate the config file using the command line argument `config_file`:

```
python3 grace/run.py --config_file="/absolute/path/to/the/config/file/config.json"
```
_e.g._
```
python3 grace/run.py --config_file=/Users/kulicna/Desktop/classifier/configs/config_hyperparams.json
```


-> rename "gcn_hidden_channels" maybe?

# Candidates for hyperparameters:

+ learning rate
+ dropout probability
+ classifier.py -> node_output_classes: int = 2 (hard-coded param)
+ classifier.py -> edge_output_classes: int = 2 (hard-coded param)
  + this should be updated from the config
+ specify which model you want to run: "gcn", "gat", "linear classifier", "central patch pixel"
+ clean the optimiser - make the inference function as `predict`
  + set the model to eval()
  + set the
