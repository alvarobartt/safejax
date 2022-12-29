from pathlib import Path
from typing import Union

from flax.core.frozen_dict import freeze
from objax.variable import VarCollection
from safetensors.flax import load, load_file

from safejax.typing import ParamsDictLike, PathLike
from safejax.utils import unflatten_dict


def deserialize(
    path_or_buf: Union[PathLike, bytes],
    freeze_dict: bool = False,
    requires_unflattening: bool = True,
    to_var_collection: bool = False,
) -> ParamsDictLike:
    """
    Deserialize JAX, Flax, Haiku, or Objax model params from either a `bytes` object or a file path,
    stored using `safetensors.flax.save_file` or directly saved using `safejax.save.serialize` with
    the `filename` parameter.

    Note:
        The default behavior of this function is to restore a `Dict[str, jnp.DeviceArray]` from
        a `bytes` object or a file path. If you are using `objax`, you should set `requires_unflattening`
        to `False` and `to_var_collection` to `True` to restore a `VarCollection`. If you're using `flax` you
        should set `freeze_dict` to `True` to restore a `FrozenDict`. Those are just tips on how to use it
        but all those frameworks are compatible with the default behavior.

    Args:
        path_or_buf:
            A `bytes` object or a file path containing the serialized model params.
        freeze_dict:
            Whether to freeze the output `Dict` to be a `FrozenDict` or not. Defaults to `False`.
        requires_unflattening:
            Whether the model params require unflattening or not. Defaults to `True`.
        to_var_collection:
            Whether to convert the output `Dict` to a `VarCollection` or not. Defaults to `False`.

    Returns:
        A `Dict[str, jnp.DeviceArray]`, `FrozenDict`, or `VarCollection` containing the model params.
    """
    if isinstance(path_or_buf, bytes):
        decoded_params = load(data=path_or_buf)
    elif isinstance(path_or_buf, (str, Path)):
        filename = path_or_buf if isinstance(path_or_buf, Path) else Path(path_or_buf)
        if not filename.exists or not filename.is_file:
            raise ValueError(
                f"`path_or_buf` must be a valid file path, not {path_or_buf}."
            )
        decoded_params = load_file(filename=filename.as_posix())
    else:
        raise ValueError(
            "`path_or_buf` must be a `bytes` object or a file path (`str` or"
            f" `pathlib.Path` object), not {type(path_or_buf)}."
        )
    if requires_unflattening:
        decoded_params = unflatten_dict(params=decoded_params)
    if freeze_dict:
        return freeze(decoded_params)
    if to_var_collection:
        return VarCollection(decoded_params)
    return decoded_params
