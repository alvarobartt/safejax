from time import perf_counter

import jax
from flax import linen as nn
from flax.serialization import to_bytes
from jax import numpy as jnp

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


start_time = perf_counter()
for _ in range(100):
    serialize(params)
end_time = perf_counter()
print(f"safejax (100 runs): {end_time - start_time:0.4f} s")

start_time = perf_counter()
for _ in range(100):
    to_bytes(params)
end_time = perf_counter()
print(f"flax (100 runs): {end_time - start_time:0.4f} s")
