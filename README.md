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

from safejax.flax import serialize


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
`safejax.flax.serialize` uses `safetensors`
* `flax.serialization.from_bytes` requires the `target` to be instantiated, while
`safejax.flax.deserialize` just needs the encoded bytes

## 🏋🏼 Benchmark

Benchmarks use [`hyperfine`](https://github.com/sharkdp/hyperfine) so it needs
to be installed first, and the `hatch`/`pyenv` environment needs to be activated
first (or just install the requirements).

```bash
$ hyperfine --warmup 2 "python benchmark.py benchmark_safejax" "python benchmark.py benchmark_flax" 
Benchmark 1: python benchmark.py benchmark_safejax
  Time (mean ± σ):     539.6 ms ±  11.9 ms    [User: 1693.2 ms, System: 690.4 ms]
  Range (min … max):   516.1 ms … 555.7 ms    10 runs
 
Benchmark 2: python benchmark.py benchmark_flax
  Time (mean ± σ):     543.2 ms ±   5.6 ms    [User: 1659.6 ms, System: 748.9 ms]
  Range (min … max):   532.0 ms … 551.5 ms    10 runs
 
Summary
  'python benchmark.py benchmark_safejax' ran
    1.01 ± 0.02 times faster than 'python benchmark.py benchmark_flax'
```

As we can see the difference is almost not noticeable, since the benchmark is using a 
2-tensor dictionary, which should be faster using any method. The main difference is on
the `safetensors` usage for the tensor storage instead of `pickle`.

More in detailed and complex benchmarks will be prepared soon!
