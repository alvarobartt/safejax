from typing import Any, Dict, Union

from flax.core.frozen_dict import FrozenDict
from safetensors.flax import save, save_file

from safejax.typing import PathLike
from safejax.utils import flatten_dict


def serialize(
    params: Union[Dict[str, Any], FrozenDict],
    filename: Union[PathLike, None] = None,
) -> Union[bytes, PathLike]:
    """
    Serialize a JAX/Flax/Haiku model params from either a `FrozenDict` or a `Dict`.

    If `filename` is not provided, the serialized model is returned as a `bytes` object,
    otherwise the model is saved to the provided `filename` and the `filename` is returned.

    Args:
        params: A `FrozenDict` or a `Dict` containing the model params.
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
