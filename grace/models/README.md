To run the model training:

```
python3 grace/run.py --config_file="/Users/kulicna/Desktop/classifier/configs/config_hyperparams_submitted.json"
python grace/run.py --config_file=/Users/kulicna/Desktop/classifier/configs/config_hyperparams_submitted_fake.json
```

absolute path to the `config_file` structured as follows:

```
{
    # Paths to data:
    "image_dir": "None",
    "grace_dir": "None",
    "log_dir": "None",
    "run_dir": "/Users/csoelistyo/Documents/grace_files", -> rename to my location

    # Feature extractor:
    "extractor_fn": "None",
    "feature_dim": "2048",

    # Patch data specs:
    "patch_size": "(224, 224)",
    "ignore_fraction": "1.0", -> required fraction of the image not to be excluded

    # Groups of augmentations:
    "img_graph_augs": "['random_edge_addition_and_removal', 'random_xy_translation']",
    "img_graph_aug_params": "[{'p_add': 0.2}, {'max_shift': 15.0}]",

    "patch_augs": "[]",
    "patch_aug_params": "[]",

    # Classifier model training:
    "hidden_channels": "32", -> rename "gcn_hidden_channels" maybe?
    "num_node_classes": "2",
    "num_edge_classes": "2",
    "epochs": "100",
    "metrics": "['accuracy', 'confusion_matrix']",
}
```

Training takes a while to start, perhaps consider including the progress logging?
