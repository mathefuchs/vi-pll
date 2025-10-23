""" Configurations. """

from glob import glob

# All real-world datasets
REAL_WORLD_DATA = list(sorted(
    glob("data/realworld-datasets/*.mat")
))
REAL_WORLD_DATA_LABELS = [
    path.split("/")[-1].split(".")[0] for path in REAL_WORLD_DATA
]
REAL_WORLD_LABEL_TO_PATH = dict(zip(REAL_WORLD_DATA_LABELS, REAL_WORLD_DATA))
