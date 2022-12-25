# 🔐 Serialize JAX/Flax models with `safetensors`

`safejax` is a Python package to serialize JAX and Flax models using `safetensors`
as the tensor storage format, instead of relying on `pickle`. For more details on why
`safetensors` is safer than `pickle` please check https://github.com/huggingface/safetensors.

## 🛠️ Requirements & Installation

`safejax` requires Python 3.7 or above

```bash
pip install safejax --upgrade
```

## 💻 Usage

```python
import jax
from flax import linen as nn
from jax import numpy as jnp

from safejax import serialize


class SingleLayerModel(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.features)(x)
        return x


model = SingleLayerModel(features=1)

rng = jax.random.PRNGKey(0)
params = model.init(rng, jnp.ones((1, 1)))

serialized = serialize(params=params)
assert isinstance(serialized, bytes)
assert len(serialized) > 0
```

More examples can be found at [`examples/`](./examples).

## 🤔 Why `safejax`?

`safetensors` defines an easy and fast (zero-copy) format to store tensors,
while `pickle` has some known weaknesses and security issues. `safetensors`
is also a storage format that is intended to be trivial to the framework
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
using `safetensors` as the tensor storage format instead of `pickle`.

### 📄 Main differences with `flax.serialization`

* `flax.serialization.to_bytes` uses `pickle` as the tensor storage format, while
`safejax.serialize` uses `safetensors`
* `flax.serialization.from_bytes` requires the `target` to be instantiated, while
`safejax.deserialize` just needs the encoded bytes

## 🏋🏼 Benchmark

Benchmarks are no longer running with [`hyperfine`](https://github.com/sharkdp/hyperfine),
as most of the elapsed time is not during the actual serialization but in the imports and
in the model parameter initialization. So we've refactored those so as to run with pure
Python code using `time.perf_counter` to measure the elapsed time in seconds.

```bash
$ python benchmarks/resnet50.py
safejax (100 runs): 2.0974 s
flax (100 runs): 4.8734 s
```

This means that for `ResNet50`, `safejax` is x2.3 times faster than `flax.serialization` when
it comes to serialization, also to restate the fact that `safejax` stores the tensors with
`safetensors` while `flax` saves those with `pickle`.

But if we use [`hyperfine`](https://github.com/sharkdp/hyperfine) as mentioned above, it needs
to be installed first, and the `hatch`/`pyenv` environment needs to be activated
first (or just install the requirements). But, due to the overhead of the script, the 
elapsed time during the serialization will be minimal compared to the rest, so the overall
result won't reflect well enough the efficiency diff between both approaches, as above.

```bash
$ hyperfine --warmup 2 "python benchmarks/hyperfine/resnet50.py serialization_safejax" "python benchmarks/hyperfine/resnet50.py serialization_flax"
Benchmark 1: python benchmarks/hyperfine/resnet50.py serialization_safejax
  Time (mean ± σ):      1.778 s ±  0.038 s    [User: 3.345 s, System: 0.511 s]
  Range (min … max):    1.741 s …  1.877 s    10 runs
 
Benchmark 2: python benchmarks/hyperfine/resnet50.py serialization_flax
  Time (mean ± σ):      1.790 s ±  0.011 s    [User: 3.371 s, System: 0.478 s]
  Range (min … max):    1.771 s …  1.810 s    10 runs
 
Summary
  'python benchmarks/hyperfine/resnet50.py serialization_safejax' ran
    1.01 ± 0.02 times faster than 'python benchmarks/hyperfine/resnet50.py serialization_flax'
```

As we can see the difference is almost not noticeable, since the benchmark is using a 
2-tensor dictionary, which should be faster using any method. The main difference is on
the `safetensors` usage for the tensor storage instead of `pickle`.
