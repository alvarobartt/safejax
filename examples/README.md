# 💻 Examples

Here you will find some detailed examples of how to use `safejax` to serialize
model parameters, in opposition to the default way to store those, which uses 
`pickle` as the format to store the tensors instead of `safetensors`.

## Flax - [`flax_ft_safejax`](./examples/flax_ft_safejax.py)

To run this Python script you won't need to install anything else than
`safejax`, as both `jax` and `flax` are installed as part of it.

In this case a single one layer model will be created, as for now, `flax`
doesn't have any pre-defined architecture such as ResNet, but you can use
[`flaxmodels`](https://github.com/matthias-wright/flaxmodels) for that, as
defines some well-known architectures written in `flax`.

```python
import jax
from flax import linen as nn

class SingleLayerModel(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.features)(x)
        return x
```

Once the network has been defined, we can instantiate and initialize it,
so as to retrieve the `params` out of the forward-pass performed during
`.init`.

```python
import jax
from jax import numpy as jnp

network = SingleLayerModel(features=1)

rng_key = jax.random.PRNGKey(seed=0)
initial_params = network.init(rng_key, jnp.ones((1, 1)))
```

Right after getting the `params` from the `.init` method's output, we can
use `safejax.serialize` to encode those using `safetensors`, that later on 
can be loaded back using `safejax.deserialize`.

```python
from safejax import deserialize, serialize

encoded_bytes = serialize(params=initial_params)
decoded_params = deserialize(path_or_buf=encoded_bytes, freeze_dict=True)
```

As seen in the code above, we're using `freeze_dict=True` since its default
value is False, as we want to freeze the dict with the params before actually
returning it during `safejax.deserialize`, this basically transforms the `Dict`
into a `FrozenDict`.

Finally, we can use those `decoded_params` so as to run a forward pass
with the previously defined single-layer network.

```python
x = jnp.ones((1, 1))
y = network.apply(decoded_params, x)
```


## Haiku - [`haiku_ft_safejax.py`](./examples/haiku_ft_safejax.py)

To run this Python script you'll need to have both `safejax` and [`dm-haiku`](https://github.com/deepmind/dm-haiku)
installed.

A ResNet50 architecture will be used from `haiku.nets.imagenet.resnet` and since
the purpose of the example is to show the integration of both `dm-haiku` and
`safejax`, we won't use pre-trained weights.

If you're not familiar with `dm-haiku`, please visit [Haiku Basics](https://dm-haiku.readthedocs.io/en/latest/notebooks/basics.html).

First of all, let's create the network instance for the ResNet50 using `dm-haiku`
with the following code:

```python
import haiku as hk
from jax import numpy as jnp

def resnet_fn(x: jnp.DeviceArray, is_training: bool):
    resnet = hk.nets.ResNet50(num_classes=10)
    return resnet(x, is_training=is_training)

network = hk.without_apply_rng(hk.transform_with_state(resnet_fn))
```

Some notes on the code above:
* `haiku.nets.ResNet50` requires `num_classes` as a mandatory parameter
* `haiku.nets.ResNet50.__call__` requires `is_training` as a mandatory parameter
* It needs to be initialized with `hk.transform_with_state` as we want to preserve
the state e.g. ExponentialMovingAverage in BatchNorm. More information at https://dm-haiku.readthedocs.io/en/latest/api.html#transform-with-state.
* Using `hk.without_apply_rng` removes the `rng` arg in the `.apply` function. Mode information at https://dm-haiku.readthedocs.io/en/latest/api.html#without-apply-rng.

Then we just initialize the network to retrieve both the `params` and the `state`,
which again, are random.

```python
import jax

rng_key = jax.random.PRNGKey(seed=0)
initial_params, initial_state = network.init(
    rng_key, jnp.ones([1, 224, 224, 3]), is_training=True
)
```

Now once we have the `params`, we can import `safejax.serialize` to serialize the 
params using `safetensors` as the tensor storage format, that later on can be loaded
back using `safejax.deserialize` and used for the network's inference.

```python
from safejax import deserialize, serialize

encoded_bytes = serialize(params=initial_params)
decoded_params = deserialize(path_or_buf=encoded_bytes)
```

Finally, let's just use those `decoded_params` to run the inference over the network
using those weights.

```python
x = jnp.ones([1, 224, 224, 3])
y, _ = network.apply(decoded_params, initial_state, x, is_training=False)
```
