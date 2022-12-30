# 🔐 Serialize JAX, Flax, Haiku, or Objax model params with `safetensors`

`safejax` is a Python package to serialize JAX, Flax, Haiku, or Objax model params using `safetensors`
as the tensor storage format, instead of relying on `pickle`. For more details on why
`safetensors` is safer than `pickle` please check [huggingface/safetensors](https://github.com/huggingface/safetensors).

Note that `safejax` supports the serialization of `jax`, `flax`, `dm-haiku`, and `objax` model
parameters and has been tested with all those frameworks, but there may be some cases where it
does not work as expected, as this is still in an early development phase, so please if you have
any feedback or bug reports, open an issue at [safejax/issues](https://github.com/alvarobartt/safejax/issues).

## 🛠️ Requirements & Installation

`safejax` requires Python 3.7 or above

```bash
pip install safejax --upgrade
```

## 💻 Usage

### `flax`

* Convert `params` to `bytes` in memory

  ```python
  from safejax import serialize, deserialize

  params = model.init(...)

  encoded_bytes = serialize(params)
  decoded_params = deserialize(encoded_bytes, freeze_dict=True)

  model.apply(decoded_params, ...)
  ```

* Convert `params` to `bytes` in `params.safetensors` file

  ```python
  from safejax import serialize, deserialize

  params = model.init(...)

  encoded_bytes = serialize(params, filename="./params.safetensors")
  decoded_params = deserialize("./params.safetensors", freeze_dict=True)

  model.apply(decoded_params, ...)
  ```

---

### `dm-haiku`

* Just contains `params`

  ```python
  from safejax import serialize, deserialize

  params = model.init(...)

  encoded_bytes = serialize(params)
  decoded_params = deserialize(encoded_bytes)

  model.apply(decoded_params, ...)
  ```

* If it contains `params` and `state` e.g. ExponentialMovingAverage in BatchNorm

  ```python
  from safejax import serialize, deserialize

  params, state = model.init(...)
  params_state = {"params": params, "state": state}
  
  encoded_bytes = serialize(params_state)
  decoded_params_state = deserialize(encoded_bytes) # .keys() contains `params` and `state`

  model.apply(decoded_params_state["params"], decoded_params_state["state"], ...)
  ```

* If it contains `params` and `state`, but we want to serialize those individually

  ```python
  from safejax import serialize, deserialize

  params, state = model.init(...)

  encoded_bytes = serialize(params)
  decoded_params = deserialize(encoded_bytes)

  encoded_bytes = serialize(state)
  decoded_state = deserialize(encoded_bytes)

  model.apply(decoded_params, decoded_state, ...)
  ```

---

### `objax`

* Convert `params` to `bytes` in memory, and convert back to `VarCollection`

  ```python
  from safejax import serialize, deserialize

  params = model.vars()

  encoded_bytes = serialize(params=params)
  decoded_params = deserialize(
    encoded_bytes, requires_unflattening=False, to_var_collection=True
  )

  for key, value in decoded_params.items():
    if key in model.vars():
      model.vars()[key].assign(value)

  model(...)
  ```

* Convert `params` to `bytes` in `params.safetensors` file

  ```python
  from safejax import serialize, deserialize

  params = model.vars()

  encoded_bytes = serialize(params=params, filename="./params.safetensors")
  decoded_params = deserialize("./params.safetensors", requires_unflattening=False)

  for key, value in decoded_params.items():
    if key in model.vars():
      model.vars()[key].assign(value)

  model(...)
  ```

* Convert `params` to `bytes` in `params.safetensors` and assign during deserialization

  ```python
  from safejax.objax import serialize, deserialize_with_assignment

  params = model.vars()

  encoded_bytes = serialize(params=params, filename="./params.safetensors")
  deserialize_with_assignment(filename="./params.safetensors", model_vars=params)

  model(...)
  ```

---

📌 As you may have seen in the examples above, most of those codeblocks are imporing both
`serialize` and `deserialize` from `safejax`, but as some of those expect params with respect
to the JAX framework that we're using, we can just import those from their files to avoid 
defining the params over and over e.g. instead of `from safejax import deserialize, serialize`,
we can just import `from safejax.flax import deserialize, serialize`, and skip the function 
params, so that the only input param that we need to provide are the params themselves.

More in-detail examples can be found at [`examples/`](./examples) for `flax`, `dm-haiku`, and `objax`.

## 🤔 Why `safejax`?

`safetensors` defines an easy and fast (zero-copy) format to store tensors,
while `pickle` has some known weaknesses and security issues. `safetensors`
is also a storage format that is intended to be trivial to the framework
used to load the tensors. More in-depth information can be found at 
[huggingface/safetensors](https://github.com/huggingface/safetensors).

`jax` uses `pytrees` to store the model parameters in memory, so
it's a dictionary-like class containing nested `jnp.DeviceArray` tensors.

`dm-haiku` uses a custom dictionary formatted as `<level_1>/~/<level_2>`, where the
levels are the ones that define the tree structure and `/~/` is the separator between those
e.g. `res_net50/~/intial_conv`, and that key does not contain a `jnp.DeviceArray`, but a 
dictionary with key value pairs e.g. for both weights as `w` and biases as `b`.

`objax` defines a custom dictionary-like class named `VarCollection` that contains
some variables inheriting from `BaseVar` which is another custom `objax` type.

`flax` defines a dictionary-like class named `FrozenDict` that is used to
store the tensors in memory, it can be dumped either into `bytes` in `MessagePack`
format or as a `state_dict`.

Of all those, `flax` is the only framework that defines its custom functions to
serialize and deserialize the model params under `flax.serialization`.But `flax` still
uses `pickle` as the format for storing the tensors, and there are no plans from HuggingFace
to extend `safetensors` to support anything more than tensors e.g. `FrozenDict`s, see their
response at [huggingface/safetensors/discussions/138](https://github.com/huggingface/safetensors/discussions/138).

So the motivation to create `safejax` is to easily provide a way to serialize `FrozenDict`s
using `safetensors` as the tensor storage format instead of `pickle`, as well as to provide
a common and easy way to serialize and deserialize any JAX model params (Flax, Haiku, or Objax)
using `safetensors` format.

### 📄 Main differences with `flax.serialization`

* `flax.serialization.to_bytes` uses `pickle` as the tensor storage format, while
`safejax.serialize` uses `safetensors`
* `flax.serialization.from_bytes` requires the `target` to be instantiated, while
`safejax.deserialize` just needs the encoded bytes

## 🏋🏼 Benchmark

Benchmarks are no longer running with [`hyperfine`](https://github.com/sharkdp/hyperfine),
as most of the elapsed time is not during the actual serialization but in the imports and
the model parameter initialization. So we've refactored those to run with pure
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
