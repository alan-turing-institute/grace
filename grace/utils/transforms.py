from typing import Callable

from grace.training.config import Config
from grace.utils.augment_graph import (
    RandomEdgeAdditionAndRemoval,
    RandomXYTranslation,
)
from grace.utils.augment_image import (
    RandomEdgeCrop,
    RandomImageGraphRotate,
)
from torchvision.transforms import Compose

PATCH_TRANSFORMS = {
    "random_edge_crop": RandomEdgeCrop,
}

GRAPH_TRANSFORMS = {
    "random_edge_addition_and_removal": RandomEdgeAdditionAndRemoval,
    "random_xy_translation": RandomXYTranslation,
    "random_image_graph_rotate": RandomImageGraphRotate,
}


class ImageGraphCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, grph):
        for t in self.transforms:
            img, grph = t(img, grph)
        return img, grph


def get_transforms(
    config: Config,
    group: str,
) -> list[Callable]:
    if group not in ["patch", "graph"]:
        raise ValueError("group must be either 'patch' or 'graph'.")

    if group == "patch":
        transform_dict = PATCH_TRANSFORMS
        strings = config.patch_augs
        params = config.patch_aug_params

        return Compose(
            [
                transform_dict[string](**params[n])
                for n, string in enumerate(strings)
            ]
        )

    elif group == "graph":
        transform_dict = GRAPH_TRANSFORMS
        strings = config.img_graph_augs
        params = config.img_graph_aug_params

        return ImageGraphCompose(
            [
                transform_dict[string](**params[n])
                for n, string in enumerate(strings)
            ]
        )
