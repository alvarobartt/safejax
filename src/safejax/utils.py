from typing import Any, Dict, Union

import numpy as np
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp
from objax.variable import BaseState, BaseVar


def flatten_dict(
    params: Union[Dict[str, Any], FrozenDict],
    key_prefix: Union[str, None] = None,
) -> Dict[str, Any]:
    """
    Flatten a `FrozenDict` or a `Dict` containing either `jnp.DeviceArray` or
    `np.ndarray` as values.

    Note:
        This function is recursive to explore all the nested dictionaries,
        and the keys are being flattened using the `.` character. So that the
        later de-nesting can be done using the `.` character as a separator.

    Reference at https://gist.github.com/Narsil/d5b0d747e5c8c299eb6d82709e480e3d

    Args:
        params: A `FrozenDict` or a `Dict` with the params to flatten.
        key_prefix: A prefix to prepend to the keys of the flattened dictionary.

    Returns:
        A `Dict` containing the flattened params.
    """
    flattened_params = {}
    for key, value in params.items():
        key = f"{key_prefix}.{key}" if key_prefix else key
        if isinstance(value, (BaseVar, BaseState)):
            value = value.value
        if isinstance(value, (jnp.DeviceArray, np.ndarray)):
            flattened_params[key] = value
            continue
        if isinstance(value, (Dict, FrozenDict)):
            flattened_params.update(
                flatten_dict(
                    params=value,
                    key_prefix=key,
                )
            )
    return flattened_params


def unflatten_dict(tensors: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unflatten a `Dict` of tensors stored as a flattened dictionary.

    Reference at https://stackoverflow.com/a/63545677.

    Args:
        tensors: A `Dict` of tensors stored as a flattened dictionary.

    Returns:
        An unflattened `Dict` of tensors.
    """
    params = {}
    for key, value in tensors.items():
        params_tmp = params
        subkeys = key.split(".")
        for subkey in subkeys[:-1]:
            params_tmp = params_tmp.setdefault(subkey, {})
        params_tmp[subkeys[-1]] = value
    return params
