from typing import Any, Callable

import torch

from torchvision.transforms import (
    Resize,
    Lambda,
    Compose,
    Normalize,
    RandomApply,
    RandomAffine,
)
from grace.utils.augment_image import RandomEdgeCrop
from grace.base import GraphAttrs


def resnet(resnet_type: str = "resnet152") -> torch.nn.Module:
    """Returns the pre-trained resnet152 model."""
    if resnet_type == "resnet152":
        from torchvision.models import resnet152, ResNet152_Weights

        classifier = resnet152(ResNet152_Weights)
    elif resnet_type == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights

        classifier = resnet50(ResNet50_Weights)
    else:
        from torchvision.models import resnet18, ResNet18_Weights

        classifier = resnet18(ResNet18_Weights)

    modules = list(classifier.children())[:-1]
    extractor = torch.nn.Sequential(*modules)
    for p in extractor.parameters():
        p.requires_grad = False

    return extractor


DEFAULT_AUGMENTATIONS = RandomApply(
    [
        RandomEdgeCrop(max_fraction=0.1),
        RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
        ),
    ],
    p=0.3,
)


class FeatureExtractor(torch.nn.Module):
    """A feature extraction model. Crops bounding boxes according to
    graph coordinates then calculates features with a specified model.

    Parameters
    ----------
    model : Callable
        Feature extractor model or function that maps a tensor
        to a tensor
    bbox_size : tuple[int]
        Size the bounding boxes to be extracted, centered
        on the x-y coordinates of each node; (...,W,H)
    normalize : bool
        Whether to run all patches through the normalize_func
    normalize_func : Callable
        Function to normalize the images before embedding extraction
    """

    def __init__(
        self,
        model: Callable,
        *,
        bbox_size: tuple[int] = (224, 224),
        normalize: bool = False,
        normalize_func: Callable = Normalize(mean=[0.0], std=[1.0]),
    ) -> None:
        super(FeatureExtractor, self).__init__()
        self.bbox_size = bbox_size
        self.model = model
        self.normalize = normalize
        self.normalize_func = normalize_func
        self.transforms = Compose(
            [
                Resize(size=bbox_size, antialias=True),
                Lambda(lambda x: x.expand(1, 3, bbox_size[0], bbox_size[1])),
            ]
        )

    def forward(
        self,
        image: torch.Tensor,
        graph: dict[str, Any],
    ) -> tuple[torch.Tensor, dict]:
        """Adds a feature vector to each graph node.

        Parameters
        ----------
        image : torch.Tensor
            Image of shape (W,H), (C,W,H) or (B,C,W,H)
        graph : dict
            Dictionary with key "graph" that accesses the networkx graph

        Returns
        -------
        image, graph : tuple[torch.Tensor, dict]
            Transformed image & graph.
        """
        # Expand image dims to 4D:
        image_shape = image.size()
        if image.ndim < 4:
            image = image[[None] * (4 - len(image.size()))]

        # Iterate through nodes:
        for _, node_attrs in graph["graph"].nodes.data():
            x, y = node_attrs[GraphAttrs.NODE_X], node_attrs[GraphAttrs.NODE_Y]

            x_low = int(x - self.bbox_size[0] / 2)
            x_box = slice(x_low, x_low + self.bbox_size[0])

            y_low = int(y - self.bbox_size[1] / 2)
            y_box = slice(y_low, y_low + self.bbox_size[1])

            # Crop the bounding box under the node:
            bbox_image = image[..., y_box, x_box]
            bbox_image = self.transforms(bbox_image)

            # Normalize if needed:
            if self.normalize is True:
                bbox_image = self.normalize_func(bbox_image)

            # Run through the feature extractor model:
            features = self.model(bbox_image)
            node_attrs[GraphAttrs.NODE_IMG_EMBEDDING] = features.squeeze()

        # Back to the original shape:
        image = image.reshape(image_shape)
        return image, graph


class SimpleDescriptor(torch.nn.Module):
    """Simple feature extractor for basic image patch characteristics.

    Parameters
    ----------
    bbox_size : tuple[int]
        Size the bounding boxes to be extracted, centered
        on the x-y coordinates of each node; (...,W,H)
    """

    def __init__(self, *, bbox_size: tuple[int] = (224, 224)) -> None:
        super(SimpleDescriptor, self).__init__()
        self.bbox_size = bbox_size

    def forward(
        self,
        image: torch.Tensor,
        graph: dict[str, Any],
    ) -> tuple[torch.Tensor, dict]:
        """Adds a feature vector to each graph node.

        Parameters
        ----------
        image : torch.Tensor
            Image of shape (W,H), (C,W,H) or (B,C,W,H)
        graph : dict
            Dictionary with key "graph" that accesses the networkx graph

        Returns
        -------
        image, graph : tuple[torch.Tensor, dict]
            Transformed image & graph.
        """
        # Expand image dims to 4D:
        image_shape = image.size()
        if image.ndim < 4:
            image = image[[None] * (4 - len(image.size()))]

        # Iterate through nodes:
        for _, node_attrs in graph["graph"].nodes.data():
            x, y = node_attrs[GraphAttrs.NODE_X], node_attrs[GraphAttrs.NODE_Y]

            x_low = int(x - self.bbox_size[0] / 2)
            x_box = slice(x_low, x_low + self.bbox_size[0])

            y_low = int(y - self.bbox_size[1] / 2)
            y_box = slice(y_low, y_low + self.bbox_size[1])

            # Crop the bounding box under the node:
            bbox_image = image[..., y_box, x_box]

            # Run through these shortlisted torch methods:
            methods = [torch.mean, torch.std, torch.min, torch.max]
            features = torch.Tensor([m(bbox_image) for m in methods])
            node_attrs[GraphAttrs.NODE_IMG_EMBEDDING] = features.squeeze()

        # Back to the original shape:
        image = image.reshape(image_shape)
        return image, graph
