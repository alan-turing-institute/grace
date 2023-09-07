from typing import Any, Dict, Tuple, Callable

import torch

from torchvision.transforms import (
    Resize,
    Lambda,
    Normalize,
    Compose,
    RandomApply,
    RandomAffine,
)

import torch.nn.functional as F

from grace.base import GraphAttrs, Annotation
from grace.utils.augment_image import RandomEdgeCrop


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


default_augmentations = RandomApply(
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
    normalise : tuple[bool]
        Whether to run all patches through the normalize_function
        before [0] and/or after [1] applying patch augmentations.
    transforms : Callable
        Series of transforms to apply to the bbox image before
        feature extraction
    augmentations : Callable
        Series of augmentations to apply to the bbox image
        during training
    normalise_func : Callable
        Function to normalize the images after the augmentations
        and after the transforms
    keep_patch_fraction : float
        Minimum fraction of either the x or y dimension of the patch that
        is missing due to boundary effects, for the patch to be kept
        (i.e., node annotation is not set to UNKNOWN)
    """

    def __init__(
        self,
        model: Callable,
        *,
        bbox_size: tuple[int] = (224, 224),
        normalise: tuple[bool] = (False, False),
        transforms: Callable = None,
        augmentations: Callable = default_augmentations,
        normalise_func: Callable = Normalize(mean=[0.0], std=[1.0]),
        keep_patch_fraction: float = 1.0,
    ) -> None:
        super(FeatureExtractor, self).__init__()
        self.bbox_size = bbox_size
        self.model = model
        self.normalise = normalise
        self.transforms = transforms
        self.augmentations = augmentations
        self.normalise_func = normalise_func
        self.keep_patch_fraction = keep_patch_fraction

        if transforms is None:
            self.transforms = Compose(
                [
                    Resize(size=bbox_size, antialias=True),
                    Lambda(
                        lambda x: x.expand(1, 3, bbox_size[0], bbox_size[1])
                    ),
                ]
            )

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

        for _, node_attrs in graph["graph"].nodes.data():
            x, y = node_attrs[GraphAttrs.NODE_X], node_attrs[GraphAttrs.NODE_Y]

            x_low = int(x - self.bbox_size[0] / 2)
            x_box = slice(x_low, x_low + self.bbox_size[0])

            y_low = int(y - self.bbox_size[1] / 2)
            y_box = slice(y_low, y_low + self.bbox_size[1])

            # Identify corner patches:
            if (
                x_low
                > image_shape[-1]
                - self.bbox_size[0] * self.keep_patch_fraction
                or x_low + self.bbox_size[0]
                < self.bbox_size[0] * self.keep_patch_fraction
                or y_low
                > image_shape[-2]
                - self.bbox_size[1] * self.keep_patch_fraction
                or y_low + self.bbox_size[0]
                < self.bbox_size[1] * self.keep_patch_fraction
            ):
                # TODO: Clarify - is this not the label by default?
                node_attrs[GraphAttrs.NODE_GROUND_TRUTH] = Annotation.UNKNOWN

            # Crop the bounding box under the node:
            bbox_image = image[..., y_box, x_box]

            x_pad_width = self.bbox_size[0] - bbox_image.size(-1)
            y_pad_width = self.bbox_size[1] - bbox_image.size(-2)

            # Pad if the patch is close to the boundary:
            if x_pad_width > 0:
                if x_low < 0:
                    x_pad = (x_pad_width, 0)
                else:
                    x_pad = (0, x_pad_width)
            else:
                x_pad = (0, 0)
            if y_pad_width > 0:
                if y_low < 0:
                    y_pad = (y_pad_width, 0)
                else:
                    y_pad = (0, y_pad_width)
            else:
                y_pad = (0, 0)

            padding = x_pad + y_pad
            bbox_image = F.pad(bbox_image, padding, "constant", 0)

            bbox_image = self.transforms(bbox_image)

            # Decide if to normalise before or after augmentations:
            norm_bef_augment, norm_aft_augment = self.normalise

            if norm_bef_augment is True:
                bbox_image = self.normalise_func(bbox_image)

            # Augment:
            if self.training:
                bbox_image = self.augmentations(bbox_image)

            if norm_aft_augment is True:
                bbox_image = self.normalise_func(bbox_image)

            # Run through the feature extractor model:
            features = self.model(bbox_image)
            node_attrs[GraphAttrs.NODE_FEATURES] = features.squeeze()

        image = image.reshape(image_shape)
        return image, graph
