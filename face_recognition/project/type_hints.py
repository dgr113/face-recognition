# coding: utf-8

import numpy as np
from pathlib import Path
from typing import Tuple, Sequence, Union, Hashable, Iterable, Mapping


UNIVERSAL_PATH_TYPE = Union[Path, str]
UNIVERSAL_SOURCE_TYPE = Union[UNIVERSAL_PATH_TYPE, Mapping]
CHUNKED_DATA_TYPE = Tuple[Sequence[np.array], Sequence[int]]
ONE_MORE_KEYS = Union[Hashable, Iterable[Hashable]]
VALIDATE_RESUTS_TYPE = Tuple[str, Union[Mapping, None]]

COORDS_TYPE = Tuple[int, int, int, int]
FRAME_SHAPE_TYPE = Tuple[int, ...]
