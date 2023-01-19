# 🤔 Why `safejax`?

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
