import warnings
from typing import Any, Dict, Union

import numpy as np
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp
from objax.variable import BaseState, BaseVar, RandomState, StateVar, TrainRef, TrainVar

from safejax.typing import JaxDeviceArrayDict, NumpyArrayDict, ObjaxDict, ParamsDictLike

OBJAX_VARIABLES = {
    "BaseVar": BaseVar,
    "BaseState": BaseState,
    "RandomState": RandomState,
    "TrainRef": TrainRef,
    "StateVar": StateVar,
    "TrainVar": TrainVar,
}
OBJAX_VARIABLE_SEPARATOR = "::"


def flatten_dict(
    params: ParamsDictLike,
    key_prefix: Union[str, None] = None,
    include_objax_variables: bool = False,
) -> Union[NumpyArrayDict, JaxDeviceArrayDict]:
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
        include_objax_variables:
            A boolean indicating whether to include the `objax.variable` types in
            the keys of the flattened dictionary.

    Returns:
        A `Dict` containing the flattened params as level-1 key-value pairs.
    """
    flattened_params = {}
    for key, value in params.items():
        key = f"{key_prefix}.{key}" if key_prefix else key
        if isinstance(value, (BaseVar, BaseState)):
            if include_objax_variables:
                key = f"{key}{OBJAX_VARIABLE_SEPARATOR}{type(value).__name__}"
            value = value.value
        if isinstance(value, (jnp.DeviceArray, np.ndarray)):
            flattened_params[key] = value
            continue
        if isinstance(value, (Dict, FrozenDict)):
            flattened_params.update(
                flatten_dict(
                    params=value,
                    key_prefix=key,
                    include_objax_variables=include_objax_variables,
                )
            )
    return flattened_params


def unflatten_dict(params: Union[NumpyArrayDict, JaxDeviceArrayDict]) -> Dict[str, Any]:
    """
    Unflatten a `Dict` where the keys should be expanded using the `.` character
    as a separator.

    Note:
        If the params where serialized from a `VarCollection` object, then the
        `objax.variable` types are included in the keys, and since this function
        just unflattens the dictionary without `objax.variable` casting, those
        variables will be ignored and unflattened normally. Anyway, when deserializing
        `objax` models you should use `safejax.objax.deserialize` or just use the
        function params in `safejax.deserialize`: `requires_unflattening=False` and
        `to_var_collection=True`.

    Reference at https://stackoverflow.com/a/63545677.

    Args:
        params: A `Dict` containing the params to unflatten by expanding the keys.

    Returns:
        An unflattened `Dict` where the keys are expanded using the `.` character.
    """
    unflattened_params = {}
    warned_user = False
    for key, value in params.items():
        unflattened_params_tmp = unflattened_params
        if not warned_user and OBJAX_VARIABLE_SEPARATOR in key:
            warnings.warn(
                "The params were serialized from a `VarCollection` object, "
                "so the `objax.variable` types are included in the keys, "
                "and since this function just unflattens the dictionary "
                "without `objax.variable` casting, those variables will be "
                "ignored and unflattened normally. Anyway, when deserializing "
                "`objax` models you should use `safejax.objax.deserialize` "
                "or just use the function params in `safejax.deserialize`: "
                "`requires_unflattening=False` and `to_var_collection=True`."
            )
            warned_user = True
        key = (
            key.split(OBJAX_VARIABLE_SEPARATOR)[0]
            if OBJAX_VARIABLE_SEPARATOR in key
            else key
        )
        subkeys = key.split(".")
        for subkey in subkeys[:-1]:
            unflattened_params_tmp = unflattened_params_tmp.setdefault(subkey, {})
        unflattened_params_tmp[subkeys[-1]] = value
    return unflattened_params


def cast_objax_variables(
    params: JaxDeviceArrayDict,
) -> Union[JaxDeviceArrayDict, ObjaxDict]:
    """
    Cast the `jnp.DeviceArray` to their corresponding `objax.variable` types.

    Note:
        This function may return the same `params` if no `objax.variable` types
        are found in the keys.

    Args:
        params: A `Dict` containing the params to cast.

    Raises:
        ValueError: If the params were not serialized from a `VarCollection` object.

    Returns:
        A `Dict` containing the keys without the variable name, and the values
        with the `objax.variable` objects with `.value` assigned from the
        `jnp.DeviceArray`.
    """
    casted_params = {}
    for key, value in params.items():
        if OBJAX_VARIABLE_SEPARATOR not in key:
            raise ValueError(
                "The params were not serialized from a `VarCollection` object, since"
                " the type has not been included as part of the key using"
                f" `{OBJAX_VARIABLE_SEPARATOR}` as separator at the end of the key."
                " Returning the same params without casting the `jnp.DeviceArray` to"
                " `objax.variable` types."
            )
        key, objax_var_type = key.split(OBJAX_VARIABLE_SEPARATOR)
        casted_params[key] = OBJAX_VARIABLES[objax_var_type](value)
    return casted_params
