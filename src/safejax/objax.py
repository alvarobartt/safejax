from functools import partial
from pathlib import Path

from objax.variable import VarCollection
from safetensors import safe_open

from safejax.core.load import deserialize
from safejax.core.save import serialize  # noqa: F401
from safejax.typing import PathLike

# `objax` expects either a `Dict[str, jnp.DeviceArray]` or a `VarCollection` as model params
# which means any other type of `Dict` will not work. This is why we need to set `requires_unflattening`
# to `False` and `to_var_collection` to `True` to restore a `VarCollection`, but the later could be skipped.
deserialize = partial(deserialize, requires_unflattening=False, to_var_collection=True)


def deserialize_with_assignment(
    filename: PathLike, model_vars: VarCollection
) -> VarCollection:
    if not isinstance(filename, (str, Path)):
        raise ValueError(
            "`filename` must be a `str` or a `pathlib.Path` object, not"
            f" {type(filename)}."
        )
    filename = filename if isinstance(filename, Path) else Path(filename)
    if not filename.exists or not filename.is_file:
        raise ValueError(f"`filename` must be a valid file path, not {filename}.")
    with safe_open(filename.as_posix(), framework="jax") as f:
        for k in f.keys():
            if k not in model_vars.keys():
                raise ValueError(f"Variable with name {k} not found in model_vars.")
            model_vars[k].assign(f.get_tensor(k))
