import os
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze
from jax import numpy as jnp
from safetensors.flax import load, load_file, save, save_file

from safejax.typing import PathLike

__all__ = ["serialize", "deserialize"]


def flatten_dict(
    params: Union[Dict[str, Any], FrozenDict],
    key_prefix: Union[str, None] = None,
) -> Union[Dict[str, jnp.DeviceArray], Dict[str, np.ndarray]]:
    """
    Flatten a `FrozenDict` or a `Dict` containing Flax model parameters.

    Note:
        This function is recursive to explore all the nested dictionaries,
        and the keys are being flattened using the `.` character. So that the
        later de-nesting can be done using the `.` character as a separator.

    Reference at https://gist.github.com/Narsil/d5b0d747e5c8c299eb6d82709e480e3d

    Args:
        params: A `FrozenDict` or a `Dict` containing the model parameters.
        key_prefix: A prefix to prepend to the keys of the flattened dictionary.

    Returns:
        A flattened dictionary containing the model parameters.
    """
    weights = {}
    for key, value in params.items():
        key = f"{key_prefix}.{key}" if key_prefix else key
        if isinstance(value, jnp.DeviceArray) or isinstance(value, np.ndarray):
            weights[key] = value
            continue
        if isinstance(value, FrozenDict) or isinstance(value, Dict):
            weights.update(flatten_dict(params=value, key_prefix=key))
    return weights


def serialize(
    params: Union[Dict[str, Any], FrozenDict],
    filename: Union[PathLike, None] = None,
) -> Union[bytes, PathLike]:
    """
    Serialize a Flax model from either a `FrozenDict` or a `Dict`.

    If `filename` is not provided, the serialized model is returned as a `bytes` object,
    otherwise the model is saved to the provided `filename` and the `filename` is returned.

    Args:
        params: A `FrozenDict` or a `Dict` containing the model parameters.
        filename: The path to the file where the model will be saved.

    Returns:
        The serialized model as a `bytes` object or the path to the file where the model was saved.
    """
    flattened_dict = flatten_dict(params=params)
    if not filename:
        return save(tensors=flattened_dict)
    else:
        save_file(tensors=flattened_dict, filename=filename)
        return filename


def unflatten_dict(tensors: Dict[str, jnp.DeviceArray]) -> FrozenDict:
    """
    Unflatten a `FrozenDict` from a `Dict` of tensors.

    Reference at https://stackoverflow.com/a/63545677.

    Args:
        tensors: A `Dict` of tensors containing the model parameters.

    Returns:
        A `FrozenDict` containing the model parameters.
    """
    weights = {}
    for key, value in tensors.items():
        subkeys = key.split(".")
        for subkey in subkeys[:-1]:
            weights = weights.setdefault(subkey, {})
        weights[subkeys[-1]] = value
    return freeze(weights)


def deserialize(path_or_buf: Union[PathLike, bytes]) -> FrozenDict:
    """
    Deserialize a Flax model from either a `bytes` object or a file path.

    Args:
        path_or_buf: A `bytes` object or a file path containing the serialized model.

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
    return unflatten_dict(tensors=loaded_dict)
