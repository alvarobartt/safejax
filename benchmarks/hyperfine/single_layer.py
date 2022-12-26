import sys

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.serialization import to_bytes

from safejax import serialize


class SingleLayerModel(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.features)(x)
        return x


model = SingleLayerModel(features=1)

rng = jax.random.PRNGKey(0)
params = model.init(rng, jnp.ones((1, 1)))


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
