# Reference from https://github.com/huggingface/datasets/blob/main/src/datasets/utils/typing.py
import os
from pathlib import Path
from typing import Dict, Union

import numpy as np
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp
from objax.variable import VarCollection

PathLike = Union[str, Path, os.PathLike]
DictionaryLike = Union[
    Dict[str, np.ndarray], Dict[str, jnp.DeviceArray], FrozenDict, VarCollection
]
