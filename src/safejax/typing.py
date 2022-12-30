from pathlib import Path
from typing import Dict, Union

from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp
from objax.variable import BaseVar, StateVar, VarCollection

PathLike = Union[str, Path]

JaxDeviceArrayDict = Dict[str, jnp.DeviceArray]
HaikuParams = Dict[str, JaxDeviceArrayDict]
ObjaxParams = Union[VarCollection, Dict[str, Union[BaseVar, StateVar]]]
FlaxParams = Union[Dict[str, Union[Dict, JaxDeviceArrayDict]], FrozenDict]

ParamsDictLike = Union[JaxDeviceArrayDict, HaikuParams, ObjaxParams, FlaxParams]
