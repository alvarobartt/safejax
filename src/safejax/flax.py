from typing import Any, Dict, Union

import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze
from jax import numpy as jnp
from safetensors.flax import load, save


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


def serialize(frozen_or_unfrozen_dict: Union[Dict[str, Any], FrozenDict]) -> bytes:
    flattened_dict = flatten_frozen_dict(
        frozen_or_unfrozen_dict=frozen_or_unfrozen_dict
    )
    return save(tensors=flattened_dict)


def unflatten_frozen_dict(tensors: Dict[str, jnp.DeviceArray]) -> FrozenDict:
    """Idea from https://stackoverflow.com/a/63545677"""
    res = {}
    for k, v in tensors.items():
        res_tmp = res
        levels = k.split(".")
        for level in levels[:-1]:
            res_tmp = res_tmp.setdefault(level, {})
        res_tmp[levels[-1]] = v
    return freeze(res)


def deserialize(data: bytes) -> FrozenDict:
    loaded_dict = load(data=data)
    return unflatten_frozen_dict(tensors=loaded_dict)
