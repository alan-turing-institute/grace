from typing import Any, Dict, Tuple, Callable

import torch

from torchvision.models import resnet152, ResNet152_Weights
from torchvision.transforms import (
    Resize,
    Lambda,
    Normalize,
    Compose,
    RandomApply,
)

from grace.base import GraphAttrs
from grace.utils.augment_image import RandomEdgeCrop


def resnet() -> torch.nn.Module:
    """Returns the pre-trained resnet152 model."""
    classifier = resnet152(ResNet152_Weights)
    modules = list(classifier.children())[:-1]
    extractor = torch.nn.Sequential(*modules)
    for p in extractor.parameters():
        p.requires_grad = False

    return extractor


default_transforms = Compose(
    [
        Resize(size=(224, 224)),
        Normalize(mean=[0.0], std=[1.0]),
        Lambda(lambda x: x.expand(1, 3, 224, 224)),
    ]
)
default_augmentations = RandomApply([RandomEdgeCrop(max_fraction=0.1)], p=0.3)


class FeatureExtractor(torch.nn.Module):
    """A feature extraction model. Crops bounding boxes according to
    graph coordinates then calculates features with a specified model.

    Parameters
    ----------
    bbox_size : Tuple[int]
        Size the bounding boxes to be extracted, centered
        on the x-y coordinates of each node; (W, H)
    model : Callable
        Feature extractor model or function that maps a tensor
        to a tensor
    transforms : Callable
        Series of transforms to apply to the bbox image before
        feature extraction
    augmentations : Callable
        Series of augmentations to apply to the bbox image
        during training
    """

    def __init__(
        self,
        model: Callable,
        *,
        bbox_size: Tuple[int] = (224, 224),
        transforms: Callable = default_transforms,
        augmentations: Callable = default_augmentations,
    ) -> None:
        super(FeatureExtractor, self).__init__()
        self.bbox_size = bbox_size
        self.model = model
        self.transforms = transforms
        self.augmentations = augmentations

    def forward(
        self,
        image: torch.Tensor,
        graph: Dict[str, Any],
    ) -> Tuple[torch.Tensor, dict]:
        """Adds a feature vector to each graph node.

        Parameters
        ----------
        image : torch.Tensor
            Image of shape (W,H), (C,W,H) or (B,C,W,H)
        graph : dict
            Dictionary with key "graph" that accesses the networkx graph
        """

        image_shape = image.size()
        if image.ndim < 4:
            image = image[[None] * (4 - len(image.size()))]

        for node_id, node_attrs in graph["graph"].nodes.data():

            x, y = node_attrs[GraphAttrs.NODE_X], node_attrs[GraphAttrs.NODE_Y]

            x_low = int(x - self.bbox_size[0] / 2)
            x_box = slice(x_low, x_low + self.bbox_size[0])

            y_low = int(y - self.bbox_size[1] / 2)
            y_box = slice(y_low, y_low + self.bbox_size[1])

            bbox_image = image[..., y_box, x_box]
            bbox_image = self.transforms(bbox_image)

            if self.training:
                bbox_image = self.augmentations(bbox_image)

            features = self.model(bbox_image)
            node_attrs[GraphAttrs.NODE_FEATURES] = features.squeeze()

        image = image.reshape(image_shape)
        return image, graph
