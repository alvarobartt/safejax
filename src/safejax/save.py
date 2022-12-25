from typing import Any, Dict, Union

import numpy as np
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp
from safetensors.flax import save, save_file

from safejax.typing import PathLike


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
    flattened_params = {}
    for key, value in params.items():
        key = f"{key_prefix}.{key}" if key_prefix else key
        if isinstance(value, jnp.DeviceArray) or isinstance(value, np.ndarray):
            flattened_params[key] = value
            continue
        if isinstance(value, FrozenDict) or isinstance(value, Dict):
            flattened_params.update(flatten_dict(params=value, key_prefix=key))
    return flattened_params


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
