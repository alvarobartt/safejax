from typing import Any, Dict, Sequence, Union


def flatten_dict(
    params: Dict[str, Any],
    key_prefix: Union[str, None] = None,
    supported_value_types: Union[Sequence[Any], None] = None,
) -> Dict[str, Any]:
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
    flattened_params = {}
    for key, value in params.items():
        key = f"{key_prefix}.{key}" if key_prefix else key
        if any(
            isinstance(value, supported_value_type)
            for supported_value_type in supported_value_types
        ):
            flattened_params[key] = value
            continue
        if isinstance(value, Dict):
            flattened_params.update(
                flatten_dict(
                    params=value,
                    key_prefix=key,
                    supported_value_types=supported_value_types,
                )
            )
    return flattened_params


def unflatten_dict(tensors: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unflatten a `FrozenDict` from a `Dict` of tensors.

    Reference at https://stackoverflow.com/a/63545677.

    Args:
        tensors: A `Dict` of tensors containing the model parameters.

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
    return params
