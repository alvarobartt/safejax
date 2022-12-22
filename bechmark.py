from time import perf_counter

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

start_time = perf_counter()
for _ in range(10000):
    to_bytes(params)
end_time = perf_counter()
to_bytes_time = end_time - start_time
print(f"to_bytes: {to_bytes_time}")

start_time = perf_counter()
for _ in range(10000):
    serialize(params)
end_time = perf_counter()
serialize_time = end_time - start_time
print(f"serialize: {serialize_time}")

if serialize_time > to_bytes_time:
    print(
        f"'to_bytes' is {to_bytes_time / serialize_time:.2f} times faster than"
        " 'serialize'"
    )
else:
    print(
        f"'serialize' is {to_bytes_time / serialize_time:.2f} times faster than"
        " 'to_bytes'"
    )
