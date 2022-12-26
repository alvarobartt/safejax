from jax import numpy as jnp
from objax.zoo.resnet_v2 import ResNet50

from safejax import deserialize, serialize

model = ResNet50(in_channels=3, num_classes=1000)

encoded_bytes = serialize(params=model.vars())
assert isinstance(encoded_bytes, bytes)
assert len(encoded_bytes) > 0

decoded_params = deserialize(
    encoded_bytes, requires_unflattening=False, to_var_collection=True
)
assert isinstance(decoded_params, dict)
assert len(decoded_params) > 0
assert decoded_params.keys() == model.vars().keys()

for key, value in decoded_params.items():
    if key not in model.vars():
        print(f"Key {key} not in model.vars()! Skipping.")
        continue
    model.vars()[key].assign(value)

x = jnp.ones((1, 3, 224, 224))
y = model(x, training=False)
assert y.shape == (1, 1000)
