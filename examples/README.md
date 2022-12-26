# 💻 Examples

Here you will find some detailed examples of how to use `safejax` to serialize
model parameters, in opposition to the default way to store those, which uses 
`pickle` as the format to store the tensors instead of `safetensors`.

## Flax - [`flax_ft_safejax`](./examples/flax_ft_safejax.py)

To run this Python script you won't need to install anything else than
`safejax`, as both `jax` and `flax` are installed as part of it.

In this case, a single-layer model will be created, for now, `flax`
doesn't have any pre-defined architecture such as ResNet, but you can use
[`flaxmodels`](https://github.com/matthias-wright/flaxmodels) for that, as
it defines some well-known architectures written in `flax`.

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
to retrieve the `params` out of the forward pass performed during
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
value is False, as we want to freeze the `dict` with the params before actually
returning it during `safejax.deserialize`, this transforms the `Dict`
into a `FrozenDict`.

Finally, we can use those `decoded_params` to run a forward pass
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
* Using `hk.without_apply_rng` removes the `rng` arg in the `.apply` function. More information at https://dm-haiku.readthedocs.io/en/latest/api.html#without-apply-rng.

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

## Objax - [`objax_ft_safejax.py`](./examples/objax_ft_safejax.py)

To run this Python script you won't need to install anything else than
`safejax`, as both `jax` and `objax` are installed as part of it.

In this case, we'll be using one of the architectures defined in the model zoo
of `objax` at [`objax/zoo`](https://github.com/google/objax/tree/master/objax/zoo),
which is ResNet50. So first of all, let's initialize it:

```python
from objax.zoo.resnet_v2 import ResNet50

model = ResNet50(in_channels=3, num_classes=1000)
```

Once initialized, we can already access the model params which in `objax` are stored
in `model.vars()` and are of type `VarCollection` which is a dictionary-like class. So
on, we can already serialize those using `safejax.serialize` and `safetensors` format
instead of `pickle` which is the current recommended way, see https://objax.readthedocs.io/en/latest/advanced/io.html.

```python
from safejax import serialize

encoded_bytes = serialize(params=model.vars())
```

Then we can just deserialize those params back using `safejax.deserialize`, and
we'll end up getting the same `VarCollection` dictionary back. Note that we need
to disable the unflattening with `requires_unflattening=False` as it's not required
due to the way it's stored, and set `to_var_collection=True` to get a `VarCollection`
instead of a `Dict[str, jnp.DeviceArray]`, even though it will work with a standard
dict too.

```python
from safejax import deserialize

decoded_params = deserialize(
    encoded_bytes, requires_unflattening=False, to_var_collection=True
)
```

Now, once decoded with `safejax.deserialize` we need to assign those key-value
pais back to the `VarCollection` of the ResNet50 via assignment, as `.update` in
`objax` has been redefined, see https://github.com/google/objax/blob/53b391bfa72dc59009c855d01b625049a35f5f1b/objax/variable.py#L311,
and it's not consistent with the standard `dict.update` (already reported at
https://github.com/google/objax/issues/254). So, instead, we need to loop over
all the key-value pairs in the decoded params and assign those one by one to the
`VarCollection` in `model.vars()`.

```python
for key, value in decoded_params.items():
    if key not in model.vars():
        print(f"Key {key} not in model.vars()! Skipping.")
        continue
    model.vars()[key].assign(value)
```

And, finally, we can run the inference over the model via the `__call__` method
as the `.vars()` are already copied from the params resulting of `safejax.deserialize`.

```python
from jax import numpy as jnp

x = jnp.ones((1, 3, 224, 224))
y = model(x, training=False)
```

Note that we're setting the `training` flag to `False`, which is the standard way
of running the inference over a pre-trained model in `objax`.
