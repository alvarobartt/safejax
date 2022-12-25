import sys

import jax
from flax.serialization import to_bytes
from flaxmodels.resnet import ResNet50
from jax import numpy as jnp

from safejax import serialize

resnet50 = ResNet50()
params = resnet50.init(jax.random.PRNGKey(42), jnp.ones((1, 224, 224, 3)))


def serialization_safejax():
    _ = serialize(params)


def serialization_flax():
    _ = to_bytes(params)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Please provide a function name to run as an argument")
    if sys.argv[1] not in globals():
        raise ValueError(f"Function {sys.argv[1]} not found")
    globals()[sys.argv[1]]()
