import os
from pathlib import Path
from typing import Union

from flax.core.frozen_dict import FrozenDict, freeze
from safetensors.flax import load, load_file

from safejax.typing import PathLike
from safejax.utils import unflatten_dict


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
        decoded_params = load(data=path_or_buf)
    if (
        isinstance(path_or_buf, str)
        or isinstance(path_or_buf, Path)
        or isinstance(path_or_buf, os.PathLike)
    ):
        decoded_params = load_file(filename=path_or_buf)
    decoded_params_dict = unflatten_dict(tensors=decoded_params)
    return freeze(decoded_params_dict) if freeze_dict else decoded_params_dict
