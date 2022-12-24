import jax
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp

from safejax.flax import deserialize, serialize


class SingleLayerModel(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.features)(x)
        return x


model = SingleLayerModel(features=1)

rng = jax.random.PRNGKey(0)
params = model.init(rng, jnp.ones((1, 1)))

serialized = serialize(frozen_or_unfrozen_dict=params)
assert isinstance(serialized, bytes)
assert len(serialized) > 0

deserialized = deserialize(path_or_buf=serialized)
assert isinstance(deserialized, FrozenDict)
assert len(deserialized) > 0
