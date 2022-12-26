from typing import Any, Dict, Union

from flax.core.frozen_dict import FrozenDict
from objax.variable import VarCollection
from safetensors.flax import save, save_file

from safejax.typing import PathLike
from safejax.utils import flatten_dict


def serialize(
    params: Union[Dict[str, Any], FrozenDict, VarCollection],
    filename: Union[PathLike, None] = None,
) -> Union[bytes, PathLike]:
    """
    Serialize JAX, Flax, Haiku, or Objax model params from either `FrozenDict`, `Dict`, or `VarCollection`.

    If `filename` is not provided, the serialized model is returned as a `bytes` object,
    otherwise the model is saved to the provided `filename` and the `filename` is returned.

    Args:
        params: A `FrozenDict`, a `Dict` or a `VarCollection` containing the model params.
        filename: The path to the file where the model params will be saved.

    Returns:
        The serialized model params as a `bytes` object or the path to the file where the model params were saved.
    """
    params = flatten_dict(params=params)
    if filename:
        save_file(tensors=params, filename=filename)
        return filename
    return save(tensors=params)
