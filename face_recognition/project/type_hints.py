# coding: utf-8

import numpy as np
import keras.utils
from pathlib import Path
from typing import Tuple, Sequence, Union, Hashable, Iterable, Mapping



### COMMON TYPES
UNIVERSAL_PATH_TYPE = Union[Path, str]
CHUNKED_DATA_TYPE = Tuple[Sequence[np.ndarray], Sequence[int]]
ONE_MORE_KEYS_TYPE = Union[Hashable, Iterable[Hashable]]
VALIDATE_RESUTS_TYPE = Tuple[str, Union[Mapping, None]]
COORDS_TYPE = Tuple[int, int, int, int]
COLOR_TYPE = Tuple[int, int, int]
KEYS_OR_NONE_TYPE = Union[Sequence[Hashable], None]
POINT_COORDS_TYPE = Tuple[int, int]


### USER INPUT DATA TYPES
UNIVERSAL_SOURCE_TYPE = Union[UNIVERSAL_PATH_TYPE, Mapping]
JSON_DATA_TYPES = Union[int, float, str, list, tuple, bool, Mapping, None]
UNIVERSAL_CONFIG_TYPE = Mapping[str, JSON_DATA_TYPES]
FRAME_SHAPE_TYPE = Tuple[int, int, int]


### LEARN MODEL TYPES
MODEL_CONFIG_TYPE = Mapping[str, Union[str, int, list, None]]
TRAIN_DATA_TYPE = Sequence[np.ndarray]
TRAIN_LABELS_TYPE = Sequence[np.ndarray]
TRAIN_DATA_GEN_TYPE = Union[Tuple[TRAIN_DATA_TYPE, TRAIN_LABELS_TYPE], keras.utils.Sequence]


### HDF5 DATA TYPES
HDF5_DATA_TYPE = Union[np.ndarray, Mapping[str, Union[int, float]]]
HDF5_GROUPED_DATA_TYPE = Mapping[str, Mapping[str, HDF5_DATA_TYPE]]
