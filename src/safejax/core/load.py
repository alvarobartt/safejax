import warnings
from pathlib import Path
from typing import Dict, Tuple, Union

from flax.core.frozen_dict import freeze
from fsspec import AbstractFileSystem
from objax.variable import VarCollection
from safetensors import safe_open
from safetensors.flax import load

from safejax.typing import ParamsDictLike, PathLike
from safejax.utils import cast_objax_variables, unflatten_dict


def deserialize(
    path_or_buf: Union[PathLike, bytes],
    fs: Union[AbstractFileSystem, None] = None,
    freeze_dict: bool = False,
    requires_unflattening: bool = True,
    to_var_collection: bool = False,
) -> Union[ParamsDictLike, Tuple[ParamsDictLike, Dict[str, str]]]:
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
        fs: The filesystem to use to load the model params. Defaults to `None`.
        freeze_dict:
            Whether to freeze the output `Dict` to be a `FrozenDict` or not. Defaults to `False`.
        requires_unflattening:
            Whether the model params require unflattening or not. Defaults to `True`.
        to_var_collection:
            Whether to convert the output `Dict` to a `VarCollection` or not. Defaults to `False`.

    Returns:
        A `Dict[str, jnp.DeviceArray]`, `FrozenDict`, or `VarCollection` containing the model params,
        or in case `path_or_buf` is a filename and `metadata` is not empty, a tuple containing the
        model params and the metadata (in that order).
    """
    metadata = {}
    if isinstance(path_or_buf, bytes):
        decoded_params = load(data=path_or_buf)
    elif isinstance(path_or_buf, (str, Path)):
        if fs and fs.protocol != "file":
            if not isinstance(fs, AbstractFileSystem):
                raise ValueError(
                    "`fs` must be a `fsspec.AbstractFileSystem` object or `None`,"
                    f" not {type(fs)}."
                )
            with fs.open(path_or_buf, "rb") as f:
                decoded_params = load(data=f.read())
        else:
            if fs and fs.protocol == "file":
                filename = Path(fs._strip_protocol(path_or_buf))
            else:
                filename = (
                    path_or_buf if isinstance(path_or_buf, Path) else Path(path_or_buf)
                )
            if not filename.exists or not filename.is_file:
                raise ValueError(
                    f"`path_or_buf` must be a valid file path, not {path_or_buf}."
                )
            decoded_params = {}
            with safe_open(filename.as_posix(), framework="jax") as f:
                metadata = f.metadata()
                for k in f.keys():
                    decoded_params[k] = f.get_tensor(k)
    else:
        raise ValueError(
            "`path_or_buf` must be a `bytes` object or a file path (`str` or"
            f" `pathlib.Path` object), not {type(path_or_buf)}."
        )
    if to_var_collection:
        try:
            return VarCollection(cast_objax_variables(params=decoded_params))
        except ValueError as e:
            warnings.warn(e)
        return decoded_params
    if requires_unflattening:
        decoded_params = unflatten_dict(params=decoded_params)
    if freeze_dict:
        return freeze(decoded_params)
    if metadata and len(metadata) > 0:
        return decoded_params, metadata
    return decoded_params
