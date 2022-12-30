from pathlib import Path
from typing import Union

from safetensors.flax import save, save_file

from safejax.typing import ParamsDictLike, PathLike
from safejax.utils import flatten_dict


def serialize(
    params: ParamsDictLike,
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
        if not isinstance(filename, (str, Path)):
            raise ValueError(
                "If `filename` is provided (not `None`), it must be a `str` or a"
                f" `pathlib.Path` object, not {type(filename)}."
            )
        filename = filename if isinstance(filename, Path) else Path(filename)
        if not filename.exists or not filename.is_file:
            raise ValueError(f"`filename` must be a valid file path, not {filename}.")
        save_file(tensors=params, filename=filename.as_posix())
        return filename

    return save(tensors=params)
