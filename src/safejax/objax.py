from functools import partial
from pathlib import Path

from objax.variable import VarCollection
from safetensors import safe_open

from safejax.core.load import deserialize
from safejax.core.save import serialize  # noqa: F401
from safejax.typing import PathLike
from safejax.utils import OBJAX_VARIABLE_SEPARATOR

# `objax` params are usually defined as a `VarCollection`, and that's basically a dictionary with
# key-value pairs where the value is either a `BaseVar` or a `StateVar`. The problem is that when
# serializing those variables by default we just keep the value which is a `jnp.DeviceArray`, so we
# need to provide `include_objax_variables=True` to store the variable type names as part of the key
# using `::` as the separator. This is useful when deserializing the params, as we can restore a
# `VarCollection` object instead of a `Dict[str, jnp.DeviceArray]`.
serialize = partial(serialize, include_objax_variables=True)

# `objax` expects either a `Dict[str, jnp.DeviceArray]` or a `VarCollection` as model params
# which means any other type of `Dict` will not work. The only difference is that `VarCollection` can
# be assigned directly to `.vars()` while `Dict[str, jnp.DeviceArray]` needs to be manually assigned
# when looping over `.vars()`. Ideally, we want to restore the params from a `VarCollection`, that's why
# we've set the `to_var_collection` parameter to `True` by default.
deserialize = partial(deserialize, requires_unflattening=False, to_var_collection=True)


# When calling `deserialize` over an `objax.variable.VarCollection` object, those variables cannot
# be used directly for the inference, as the forward pass in `objax` is done through `__call__`, which
# implies that the class instance must contain the model params loaded in `.vars` attribute. So this
# function has been created in order to ease the parameter loading for `objax`, since as opposed to
# `flax` and `haiku`, the model params are not provided on every forward pass.
def deserialize_with_assignment(filename: PathLike, model_vars: VarCollection) -> None:
    """Deserialize a `VarCollection` from a file and assign it to a `VarCollection` object.

    Note:
        This function avoid some known issues related to the variable deserialization with `objax`,
        since the params are stored in a `VarCollection` object, which contains some `objax.variable`
        variables instead of key-tensor pais. So this way we avoid having to restore the `objax.variable`
        type per each value.

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
        for key in f.keys():
            just_key = (
                key.split(OBJAX_VARIABLE_SEPARATOR)[0]
                if OBJAX_VARIABLE_SEPARATOR in key
                else key
            )
            if just_key not in model_vars.keys():
                raise ValueError(f"Variable with name {key} not found in model_vars.")
            model_vars[just_key].assign(f.get_tensor(key))
