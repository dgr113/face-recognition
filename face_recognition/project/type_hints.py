# coding: utf-8

import numpy as np
from pathlib import Path
from typing import Tuple, Sequence, Union, Hashable, Iterable, Mapping, Any


### COMMON TYPES
UNIVERSAL_PATH_TYPE = Union[Path, str]
UNIVERSAL_SOURCE_TYPE = Union[UNIVERSAL_PATH_TYPE, Mapping]
CHUNKED_DATA_TYPE = Tuple[Sequence[np.ndarray], Sequence[int]]
ONE_MORE_KEYS = Union[Hashable, Iterable[Hashable]]
VALIDATE_RESUTS_TYPE = Tuple[str, Union[Mapping, None]]
COORDS_TYPE = Tuple[int, int, int, int]
KEYS_OR_NONE_TYPE = Union[Sequence[Hashable], None]


### USER INPUT DATA TYPES
PERSONS_DATA_TYPE = Mapping[str, Mapping[str, str]]
CAMERA_DATA_TYPE = Mapping[str, Any]
FRAME_SHAPE_TYPE = Tuple[int, int, int]


### LEARN MODEL TYPES
MODEL_CONFIG_TYPE = Mapping[str, Union[str, int, list, None]]
TRAIN_DATA_TYPE = Sequence[np.ndarray]
TRAIN_LABELS_TYPE = Sequence[np.ndarray]
TRAIN_DATA_GEN_TYPE = Tuple[TRAIN_DATA_TYPE, TRAIN_LABELS_TYPE]
