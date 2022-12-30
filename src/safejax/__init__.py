"""`safejax `: Serialize JAX, Flax, Haiku, or Objax model params with `safetensors`"""

__author__ = "Alvaro Bartolome <alvarobartt@yahoo.com>"
__version__ = "0.4.0"

from safejax.core.load import deserialize  # noqa: F401
from safejax.core.save import serialize  # noqa: F401
