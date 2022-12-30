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


# When calling `deserialize` over an `objax.variable.VarCollection` object, those variables cannot
# be used directly for the inference, as the forward pass in `objax` is done through `__call__`, which
# implies that the class instance must contain the model params loaded in `.vars` attribute. So this
# function has been created in order to ease the parameter loading for `objax`, since as opposed to
# `flax` and `haiku`, the model params are not provided on every forward pass.
def deserialize_with_assignment(filename: PathLike, model_vars: VarCollection) -> None:
    """Deserialize a `VarCollection` from a file and assign it to a `VarCollection` object.

    Args:
        filename: Path to the file containing the serialized `VarCollection` as a `Dict[str, jnp.DeviceArray]`.
        model_vars: `VarCollection` object to which the deserialized tensors will be assigned.

    Returns:
        `None`, as the deserialized tensors are assigned to the `model_vars` object. So you
        just need to access `model_vars`, or the actual `model.vars()` attribute, since the
        assignment is done over a class attribute named `vars`.
    """
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
