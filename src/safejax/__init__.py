"""`safejax `: Serialize JAX/Flax models with `safetensors`"""

__author__ = "Alvaro Bartolome <alvarobartt@yahoo.com>"
__version__ = "0.2.0.dev0"

from safejax.load import deserialize  # noqa: F401
from safejax.save import serialize  # noqa: F401
