from pathlib import Path
from typing import Dict, Union

import numpy as np
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp
from objax.variable import BaseVar, StateVar, VarCollection

PathLike = Union[str, Path]

NumpyArrayDict = Dict[str, np.ndarray]
JaxDeviceArrayDict = Dict[str, jnp.DeviceArray]
HaikuParams = Dict[str, JaxDeviceArrayDict]
ObjaxDict = Dict[str, Union[BaseVar, StateVar]]
ObjaxParams = Union[VarCollection, ObjaxDict]
FlaxParams = Union[Dict[str, Union[Dict, JaxDeviceArrayDict]], FrozenDict]

ParamsDictLike = Union[JaxDeviceArrayDict, HaikuParams, ObjaxParams, FlaxParams]
