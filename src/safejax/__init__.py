"""`safejax `: Serialize JAX, Flax, Haiku, or Objax model params with `safetensors`"""

__author__ = "Alvaro Bartolome <alvarobartt@yahoo.com>"
__version__ = "0.3.0"

from safejax.load import deserialize  # noqa: F401
from safejax.save import serialize  # noqa: F401
