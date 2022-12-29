from functools import partial

from safejax.core.load import deserialize
from safejax.core.save import serialize  # noqa: F401

# `flax` expects either a `Dict[str, Any` or a `FrozenDict`, but for robustness we are
# setting `freeze_dict` to `True` to restore a `FrozenDict` which contains the params
# frozen to avoid any accidental mutation.
deserialize = partial(deserialize, freeze_dict=True)
