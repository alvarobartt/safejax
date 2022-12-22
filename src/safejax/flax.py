from typing import Any, Dict, Union

import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze
from jax import numpy as jnp
from safetensors.flax import load, load_file, save, save_file

from safejax.typing import PathLike


def flatten_frozen_dict(
    frozen_or_unfrozen_dict: Union[Dict[str, Any], FrozenDict],
    key_prefix: Union[str, None] = None,
) -> Union[Dict[str, jnp.DeviceArray], Dict[str, np.ndarray]]:
    """Idea from https://gist.github.com/Narsil/d5b0d747e5c8c299eb6d82709e480e3d"""
    weights = {}
    for key, value in frozen_or_unfrozen_dict.items():
        key = f"{key_prefix}.{key}" if key_prefix else key
        if isinstance(value, jnp.DeviceArray) or isinstance(value, np.ndarray):
            weights[key] = value
            continue
        if isinstance(value, FrozenDict) or isinstance(value, Dict):
            weights.update(
                flatten_frozen_dict(frozen_or_unfrozen_dict=value, key_prefix=key)
            )
    return weights


def serialize(
    frozen_or_unfrozen_dict: Union[Dict[str, Any], FrozenDict],
    filename: Union[PathLike, None] = None,
) -> bytes:
    # TODO(alvaro): handle the errors properly
    flattened_dict = flatten_frozen_dict(
        frozen_or_unfrozen_dict=frozen_or_unfrozen_dict
    )
    return (
        save(tensors=flattened_dict)
        if not filename
        else save_file(tensors=flattened_dict, filename=filename)
    )


def unflatten_frozen_dict(tensors: Dict[str, jnp.DeviceArray]) -> FrozenDict:
    """Idea from https://stackoverflow.com/a/63545677"""
    weights = {}
    for key, value in tensors.items():
        subkeys = key.split(".")
        for subkey in subkeys[:-1]:
            weights = weights.setdefault(subkey, {})
        weights[subkeys[-1]] = value
    return freeze(weights)


def deserialize(path_or_buf: Union[PathLike, bytes]) -> FrozenDict:
    # TODO(alvaro): handle the errors properly
    if isinstance(path_or_buf, bytes):
        loaded_dict = load(data=path_or_buf)
    if isinstance(path_or_buf, PathLike):
        loaded_dict = load_file(filename=path_or_buf)
    return unflatten_frozen_dict(tensors=loaded_dict)
