import jax
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from flax.serialization import from_bytes, to_bytes
from jax import numpy as jnp


class SingleLayerModel(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.features)(x)
        return x


model = SingleLayerModel(features=1)

rng = jax.random.PRNGKey(0)
params = model.init(rng, jnp.ones((1, 1)))

serialized = to_bytes(target=params)
assert isinstance(serialized, bytes)
assert len(serialized) > 0

deserialized = from_bytes(target=model, encoded_bytes=serialized)
assert isinstance(deserialized, dict)
assert len(deserialized) > 0
