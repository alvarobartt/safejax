import sys

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.serialization import to_bytes

from safejax.flax import serialize


class SingleLayerModel(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.features)(x)
        return x


model = SingleLayerModel(features=1)

rng = jax.random.PRNGKey(0)
params = model.init(rng, jnp.ones((1, 1)))


def benchmark_safejax():
    _ = serialize(params)


def benchmark_flax():
    _ = to_bytes(params)


if __name__ == "__main__":
    globals()[sys.argv[1]]()
