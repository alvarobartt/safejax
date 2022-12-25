import os
from pathlib import Path
from typing import Dict, Union

from flax.core.frozen_dict import FrozenDict, freeze
from jax import numpy as jnp
from safetensors.flax import load, load_file

from safejax.typing import PathLike


def unflatten_dict(
    tensors: Dict[str, jnp.DeviceArray], freeze_dict: bool = False
) -> FrozenDict:
    """
    Unflatten a `FrozenDict` from a `Dict` of tensors.

    Reference at https://stackoverflow.com/a/63545677.

    Args:
        tensors: A `Dict` of tensors containing the model parameters.
        freeze_dict: Whether to freeze the `Dict` to be a `FrozenDict` or not.

    Returns:
        A `FrozenDict` containing the model parameters.
    """
    params = {}
    for key, value in tensors.items():
        params_tmp = params
        subkeys = key.split(".")
        for subkey in subkeys[:-1]:
            params_tmp = params_tmp.setdefault(subkey, {})
        params_tmp[subkeys[-1]] = value
    return freeze(params) if freeze_dict else params


def deserialize(
    path_or_buf: Union[PathLike, bytes], freeze_dict: bool = False
) -> FrozenDict:
    """
    Deserialize a Flax model from either a `bytes` object or a file path.

    Args:
        path_or_buf: A `bytes` object or a file path containing the serialized model.
        freeze_dict: Whether to freeze the `Dict` to be a `FrozenDict` or not.

    Returns:
        A `FrozenDict` containing the model parameters.
    """
    if isinstance(path_or_buf, bytes):
        loaded_dict = load(data=path_or_buf)
    if (
        isinstance(path_or_buf, str)
        or isinstance(path_or_buf, Path)
        or isinstance(path_or_buf, os.PathLike)
    ):
        loaded_dict = load_file(filename=path_or_buf)
    return unflatten_dict(tensors=loaded_dict, freeze_dict=freeze_dict)
