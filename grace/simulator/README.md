# Creating a synthetic dataset with shapes

## Drawing geometric shapes onto a random graph:

This README.md contains instructions on how to create a synthetic dataset with random geometric shapes (currently squares, circles and stars) drown around each node of a random graph.

The random graph has the noise nodes set as `true_negatives (TN)`, whereas all the nodes which are part of an object are marked as `true_positives (TP)`.

Therefore, when drawing these geometric objects, all `TN` nodes will be placed on patches marked with 0.0 < values < 0.5, whilst all `TP` nodes will be marked with 0.5 < values < 1.0.


## Creating a dataset to benchmark GRACE:

To create a synthetic dataset, activate your environment and run:

```sh
python grace/simulator/simulate_dataset.py
```

The script has a few pre-set hyperparameters which can be changed:

```sh
VALUE = 0.5   # background value of the image canvas
SCALE = 3500   # whole image shape assuming a square shape
NUM_IMAGES = 10   # number of new images to synthesize
PATCH_SIZE = 224   # size of the patch to modify under each node
MOTIF = "lines"   # type of geometric object to draw (extends to 'circles', 'curves', 'spiral' or 'mixed')
DRAWING = "squares"   # type of modification of the patch (extends to 'circles' or 'stars')
PADDING = (112, 112)   # padding of the image in case boundary nodes & their patches need to be modified, too - otherwise nodes lying too close to the boundary will be left untouched
```

Happy simulating :-)
