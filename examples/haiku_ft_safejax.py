import haiku as hk
import jax
from jax import numpy as jnp

from safejax import deserialize, serialize


def resnet_fn(x: jnp.DeviceArray, is_training: bool):
    resnet = hk.nets.ResNet50(num_classes=10)
    return resnet(x, is_training=is_training)


network = hk.without_apply_rng(hk.transform_with_state(resnet_fn))

rng_key = jax.random.PRNGKey(seed=0)
initial_params, initial_state = network.init(
    rng_key, jnp.ones([1, 224, 224, 3]), is_training=True
)

encoded_bytes = serialize(params=initial_params)
assert isinstance(encoded_bytes, bytes)
assert len(encoded_bytes) > 0

decoded_params = deserialize(path_or_buf=encoded_bytes)
assert isinstance(decoded_params, dict)
assert len(decoded_params) > 0
assert decoded_params.keys() == initial_params.keys()

x = jnp.ones([1, 224, 224, 3])
y, _ = network.apply(decoded_params, initial_state, x, is_training=False)
assert y.shape == (1, 10)
