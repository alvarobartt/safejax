from typing import Any, Dict, Union

import numpy as np
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp
from objax.variable import BaseState, BaseVar

from safejax.typing import ParamsDictLike


def flatten_dict(
    params: ParamsDictLike,
    key_prefix: Union[str, None] = None,
) -> Union[Dict[str, np.ndarray], Dict[str, jnp.DeviceArray]]:
    """
    Flatten a `Dict`, `FrozenDict`, or `VarCollection`, for more detailed information on
    the supported input types check `safejax.typing.ParamsDictLike`.

    Note:
        This function is recursive to explore all the nested dictionaries,
        and the keys are being flattened using the `.` character. So that the
        later de-nesting can be done using the `.` character as a separator.

    Reference at https://gist.github.com/Narsil/d5b0d747e5c8c299eb6d82709e480e3d

    Args:
        params: A `Dict`, `FrozenDict`, or `VarCollection` with the params to flatten.
        key_prefix: A prefix to prepend to the keys of the flattened dictionary.

    Returns:
        A `Dict` containing the flattened params as level-1 key-value pairs.
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


# Note that this function has a less restrictive type hint than the `flatten_dict` function.
# This is because the `unflatten_dict` function is generic, and it can be used to unflatten
# any `Dict` where the keys are expanded using the `.` character as a separator.
def unflatten_dict(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unflatten a `Dict` where the keys should be expanded using the `.` character
    as a separator.

    Reference at https://stackoverflow.com/a/63545677.

    Args:
        params: A `Dict` containing the params to unflatten by expanding the keys.

    Returns:
        An unflattened `Dict` where the keys are expanded using the `.` character.
    """
    unflattened_params = {}
    for key, value in params.items():
        unflattened_params_tmp = unflattened_params
        subkeys = key.split(".")
        for subkey in subkeys[:-1]:
            unflattened_params_tmp = unflattened_params_tmp.setdefault(subkey, {})
        unflattened_params_tmp[subkeys[-1]] = value
    return unflattened_params
