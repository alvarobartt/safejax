import os
import tempfile
from pathlib import Path
from typing import Union

from fsspec import AbstractFileSystem
from safetensors.flax import save, save_file

from safejax.typing import ParamsDictLike, PathLike
from safejax.utils import flatten_dict


def serialize(
    params: ParamsDictLike,
    include_objax_variables: bool = False,
    filename: Union[PathLike, None] = None,
    fs: Union[AbstractFileSystem, None] = None,
) -> Union[bytes, PathLike]:
    """
    Serialize JAX, Flax, Haiku, or Objax model params from either `FrozenDict`, `Dict`, or `VarCollection`.

    If `filename` is not provided, the serialized model is returned as a `bytes` object,
    otherwise the model is saved to the provided `filename` and the `filename` is returned.

    Args:
        params: A `FrozenDict`, a `Dict` or a `VarCollection` containing the model params.
        include_objax_variables: Whether to include `objax.Variable` objects in the serialized model params.
        filename: The path to the file where the model params will be saved.
        fs: The filesystem to use to save the model params. Defaults to `None`.

    Returns:
        The serialized model params as a `bytes` object or the path to the file where the model params were saved.
    """
    params = flatten_dict(
        params=params, include_objax_variables=include_objax_variables
    )

    if filename:
        if not isinstance(filename, (str, Path)):
            raise ValueError(
                "If `filename` is provided (not `None`), it must be a `str` or a"
                f" `pathlib.Path` object, not {type(filename)}."
            )
        if fs and fs.protocol != "file":
            if not isinstance(fs, AbstractFileSystem):
                raise ValueError(
                    "`fs` must be a `fsspec.AbstractFileSystem` object or `None`,"
                    f" not {type(fs)}."
                )
            temp_filename = tempfile.NamedTemporaryFile(
                mode="wb", suffix=".safetensors", delete=False
            )
            try:
                temp_filename.write(save(tensors=params))
            finally:
                temp_filename.close()
                fs.put_file(lpath=temp_filename.name, rpath=filename)
                os.remove(temp_filename.name)
        else:
            if fs and fs.protocol == "file":
                filename = Path(fs._strip_protocol(filename))
            else:
                filename = filename if isinstance(filename, Path) else Path(filename)
            if not filename.exists or not filename.is_file:
                raise ValueError(
                    f"`filename` must be a valid file path, not {filename}."
                )
            save_file(tensors=params, filename=filename.as_posix())
        return filename

    return save(tensors=params)
