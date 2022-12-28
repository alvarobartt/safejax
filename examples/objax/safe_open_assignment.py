import tempfile
from copy import deepcopy

from jax import numpy as jnp
from objax.zoo.resnet_v2 import ResNet50
from safetensors import safe_open

from safejax import serialize

model = ResNet50(in_channels=3, num_classes=1000)

with tempfile.TemporaryDirectory() as tmpdirname:
    filename = serialize(
        params=model.vars(), filename=f"{tmpdirname}/params.safetensors"
    )

    # We copy the weights of the first Conv2D layer, then zero them out. This is done
    # to ensure that the weights are properly loaded from the safetensors file.
    conv2d_w = deepcopy(model.vars()["(ResNet50)[0](Conv2D).w"].value)
    model.vars()["(ResNet50)[0](Conv2D).w"].assign(
        jnp.zeros((7, 7, 3, 64), dtype=jnp.float32)
    )

    assert jnp.array_equal(
        model.vars()["(ResNet50)[0](Conv2D).w"].value,
        jnp.zeros((7, 7, 3, 64), dtype=jnp.float32),
    )

    with safe_open(filename, framework="flax") as f:
        for k in f.keys():
            if k in model.vars():
                model.vars()[k].assign(f.get_tensor(k))

    assert jnp.array_equal(model.vars()["(ResNet50)[0](Conv2D).w"].value, conv2d_w)

x = jnp.ones((1, 3, 224, 224))
y = model(x, training=False)
assert y.shape == (1, 1000)
