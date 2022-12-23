# 🧪 `safejax`

`safejax` is a Python package to serialize JAX and Flax models using `safetensors`
as the tensor storage format, instead of relying on `pickle`. For more details on why
`safetensors` is safer than `pickle` please check https://github.com/huggingface/safetensors.

## 🛠️ Requirements & Installation

`safejax` requires Python 3.7 or above, up until Python 3.11.

```bash
pip install safejax --upgrade
```

## 💻 Usage

```python
import jax

from flax import linen as nn
from flax.core.frozen_dict import FrozenDict

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
```

More examples can be found at [`examples/`](./examples).

## 🤔 Why `safejax`?

`safetensors` defines an easy and fast (zero-copy) format to store tensors,
while `pickle` has some known weaknesses and security issues. `safetensors`
is also a storage-format that is intended to be trivial to the framework
used to load the tensors. More in depth information can be found at 
https://github.com/huggingface/safetensors.

`flax` defines a dictionary-like class named `FrozenDict` that is used to
store the tensors in memory, it can be dumped either into `bytes` in `MessagePack`
format or as a `state_dict`.

Anyway, `flax` still uses `pickle` as the format for storing the tensors, so 
there are no plans from HuggingFace to extend `safetensors` to support anything
more than tensors e.g. `FrozenDict`s, see their response at
https://github.com/huggingface/safetensors/discussions/138.

So `safejax` was created so as to easily provide a way to serialize `FrozenDict`s
using `safetensors` as the tensor storage-format instead of `pickle`.
