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
  from safejax.flax import serialize, deserialize

  params = model.init(...)

  encoded_bytes = serialize(params)
  decoded_params = deserialize(encoded_bytes)

  model.apply(decoded_params, ...)
  ```

* Convert `params` to `bytes` in `params.safetensors` file

  ```python
  from safejax.flax import serialize, deserialize

  params = model.init(...)

  encoded_bytes = serialize(params, filename="./params.safetensors")
  decoded_params = deserialize("./params.safetensors")

  model.apply(decoded_params, ...)
  ```

---

### `dm-haiku`

* Just contains `params`

  ```python
  from safejax.haiku import serialize, deserialize

  params = model.init(...)

  encoded_bytes = serialize(params)
  decoded_params = deserialize(encoded_bytes)

  model.apply(decoded_params, ...)
  ```

* If it contains `params` and `state` e.g. ExponentialMovingAverage in BatchNorm

  ```python
  from safejax.haiku import serialize, deserialize

  params, state = model.init(...)
  params_state = {"params": params, "state": state}
  
  encoded_bytes = serialize(params_state)
  decoded_params_state = deserialize(encoded_bytes) # .keys() contains `params` and `state`

  model.apply(decoded_params_state["params"], decoded_params_state["state"], ...)
  ```

* If it contains `params` and `state`, but we want to serialize those individually

  ```python
  from safejax.haiku import serialize, deserialize

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
  from safejax.objax import serialize, deserialize

  params = model.vars()

  encoded_bytes = serialize(params=params)
  decoded_params = deserialize(encoded_bytes)

  for key, value in decoded_params.items():
    if key in model.vars():
      model.vars()[key].assign(value.value)

  model(...)
  ```

* Convert `params` to `bytes` in `params.safetensors` file

  ```python
  from safejax.objax import serialize, deserialize

  params = model.vars()

  encoded_bytes = serialize(params=params, filename="./params.safetensors")
  decoded_params = deserialize("./params.safetensors")

  for key, value in decoded_params.items():
    if key in model.vars():
      model.vars()[key].assign(value.value)

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

There are no plans from HuggingFace to extend `safetensors` to support anything more than tensors
e.g. `FrozenDict`s, see their response at [huggingface/safetensors/discussions/138](https://github.com/huggingface/safetensors/discussions/138).

So the motivation to create `safejax` is to easily provide a way to serialize `FrozenDict`s
using `safetensors` as the tensor storage format instead of `pickle`, as well as to provide
a common and easy way to serialize and deserialize any JAX model params (Flax, Haiku, or Objax)
using `safetensors` format.
