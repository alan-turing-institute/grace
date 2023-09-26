import logging

# import warnings
# import matplotlib.pyplot as plt


LOGGER = logging
LOGGER.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)

# warnings.filterwarnings('ignore')

# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 22}

# matplotlib.rc('font', **font)

# FONT_SETTER = plt.rcParams["font.size"] = 12  # desired font size
# LOGGER.info(f"Font set to: {FONT_SETTER}")


COLORMAPS = {
    "annotation": "turbo",
    "classifier": "coolwarm",
    "conf_matrix": "copper",
    "manifold": "rainbow",
    "patches": "binary_r",
    "mask": "binary_r",
}