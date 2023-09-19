import logging
import matplotlib.pyplot as plt

LOGGER = logging
LOGGER.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)

# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 22}

# matplotlib.rc('font', **font)

FONT_SETTER = plt.rcParams["font.size"] = 12  # desired font size
LOGGER.info(f"Font set to: {FONT_SETTER}")
