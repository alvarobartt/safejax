import jax
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp

from safejax import deserialize, serialize


class SingleLayerModel(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.features)(x)
        return x


network = SingleLayerModel(features=1)

rng_key = jax.random.PRNGKey(seed=0)
initial_params = network.init(rng_key, jnp.ones((1, 1)))

encoded_bytes = serialize(params=initial_params)
assert isinstance(encoded_bytes, bytes)
assert len(encoded_bytes) > 0

decoded_params = deserialize(path_or_buf=encoded_bytes, freeze_dict=True)
assert isinstance(decoded_params, FrozenDict)
assert len(decoded_params) > 0
assert decoded_params.keys() == initial_params.keys()

x = jnp.ones((1, 1))
y = network.apply(decoded_params, x)
assert y.shape == (1, 1)
