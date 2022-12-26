from time import perf_counter

import jax
from flax.serialization import to_bytes
from flaxmodels.resnet import ResNet50
from jax import numpy as jnp

from safejax import serialize

resnet50 = ResNet50()
params = resnet50.init(jax.random.PRNGKey(42), jnp.ones((1, 224, 224, 3)))


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
