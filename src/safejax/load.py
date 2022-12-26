import os
from pathlib import Path
from typing import Dict, Union

from flax.core.frozen_dict import FrozenDict, freeze
from jax import numpy as jnp
from objax.variable import VarCollection
from safetensors.flax import load, load_file

from safejax.typing import PathLike
from safejax.utils import unflatten_dict


def deserialize(
    path_or_buf: Union[PathLike, bytes],
    freeze_dict: bool = False,
    requires_unflattening: bool = True,
    to_var_collection: bool = False,
) -> Union[FrozenDict, Dict[str, jnp.DeviceArray], VarCollection]:
    """
    Deserialize JAX, Flax, Haiku, or Objax model params from either a `bytes` object or a file path,
    stored using `safetensors.flax.save_file` or directly saved using `safejax.save.serialize` with
    the `filename` parameter.

    Note:
        The default behavior of this function is to restore a `Dict[str, jnp.DeviceArray]` from
        a `bytes` object or a file path. If you are using `objax`, you should set `requires_unflattening`
        to `False` and `to_var_collection` to `True` to restore a `VarCollection`. If you're using `flax` you
        should set `freeze_dict` to `True` to restore a `FrozenDict`. Those are just tips on how to use it
        but all those frameworks are compatible with the default behavior.

    Args:
        path_or_buf:
            A `bytes` object or a file path containing the serialized model params.
        freeze_dict:
            Whether to freeze the output `Dict` to be a `FrozenDict` or not. Defaults to `False`.
        requires_unflattening:
            Whether the model params require unflattening or not. Defaults to `True`.
        to_var_collection:
            Whether to convert the output `Dict` to a `VarCollection` or not. Defaults to `False`.

    Returns:
        A `Dict[str, jnp.DeviceArray]`, `FrozenDict`, or `VarCollection` containing the model params.
    """
    if isinstance(path_or_buf, bytes):
        decoded_params = load(data=path_or_buf)
    if (
        isinstance(path_or_buf, str)
        or isinstance(path_or_buf, Path)
        or isinstance(path_or_buf, os.PathLike)
    ):
        decoded_params = load_file(filename=path_or_buf)
    if requires_unflattening:
        decoded_params = unflatten_dict(params=decoded_params)
    if freeze_dict:
        return freeze(decoded_params)
    if to_var_collection:
        return VarCollection(decoded_params)
    return decoded_params
