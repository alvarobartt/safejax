from functools import partial

from safejax.core.load import deserialize
from safejax.core.save import serialize  # noqa: F401

# `objax` expects either a `Dict[str, jnp.DeviceArray]` or a `VarCollection` as model params
# which means any other type of `Dict` will not work. This is why we need to set `requires_unflattening`
# to `False` and `to_var_collection` to `True` to restore a `VarCollection`, but the later could be skipped.
deserialize = partial(deserialize, requires_unflattening=False, to_var_collection=True)
