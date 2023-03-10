# 💻 Usage

## `flax`

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

## `dm-haiku`

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

## `objax`

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

More in-detail examples can be found at [`examples/`](https://github.com/alvarobartt/safejax/examples)
for `flax`, `dm-haiku`, and `objax`.
