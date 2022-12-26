# 🤔 Why `safejax`?

`safetensors` defines an easy and fast (zero-copy) format to store tensors,
while `pickle` has some known weaknesses and security issues. `safetensors`
is also a storage format that is intended to be trivial to the framework
used to load the tensors. More in-depth information can be found at 
https://github.com/huggingface/safetensors.

Both `jax` and `haiku` use `pytrees` to store the model parameters in memory, so
it's a dictionary-like class containing nested `jnp.DeviceArray` tensors.

Then `objax` defines a custom dictionary-like class named `VarCollection` that contains
some variables inheriting from `BaseVar` which is another custom `objax` type.

`flax` defines a dictionary-like class named `FrozenDict` that is used to
store the tensors in memory, it can be dumped either into `bytes` in `MessagePack`
format or as a `state_dict`.

Anyway, `flax` still uses `pickle` as the format for storing the tensors, so 
there are no plans from HuggingFace to extend `safetensors` to support anything
more than tensors e.g. `FrozenDict`s, see their response at
https://github.com/huggingface/safetensors/discussions/138.

So `safejax` was created to easily provide a way to serialize `FrozenDict`s
using `safetensors` as the tensor storage format instead of `pickle`.

## 📄 Main differences with `flax.serialization`

* `flax.serialization.to_bytes` uses `pickle` as the tensor storage format, while
`safejax.serialize` uses `safetensors`
* `flax.serialization.from_bytes` requires the `target` to be instantiated, while
`safejax.deserialize` just needs the encoded bytes
