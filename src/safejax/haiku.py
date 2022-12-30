from safejax.core.load import deserialize  # noqa: F401
from safejax.core.save import serialize  # noqa: F401

# Nothing here as `dm-haiku` works with the default behavior of both
# `safejax.core.load.deserialize` and `safejax.core.save.serialize`. But
# placing this here for consistency with the other frameworks.
