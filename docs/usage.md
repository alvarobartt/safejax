# 💻 Usage

Let's create a `flax` model using the Linen API and initialize it.

```python
import jax
from flax import linen as nn
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
```

Once done, we can already save the model params with `safejax` (using `safetensors`
storage format) using `safejax.serialize`.

```python
from safejax import serialize

serialized_params = serialize(params=params)
```

Those params can be later loaded using `safejax.deserialize` and used
to run the inference over the model using those weights.

```python
from safejax import deserialize

params = deserialize(path_or_buf=serialized_params, freeze_dict=True)
```

And, finally, running the inference as:

```python
x = jnp.ones((1, 28, 28, 1))
y = model.apply(params, x)
```

More in-detail examples can be found at [`examples/`](./examples) for `flax`, `dm-haiku`, and `objax`.
