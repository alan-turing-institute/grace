import logging
import seaborn as sn  # noqa: F401


LOGGER = logging
LOGGER.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)

COLORMAPS = {
    "annotation": "turbo",
    "classifier": "coolwarm",
    "conf_matrix": "copper",
    "manifold": "rainbow",
    "patches": "binary_r",
    "mask": "binary_r",
}

sn.set_theme(
    context="notebook",
    style="dark",
    font="Helvetica",
    font_scale=1.5,
)
