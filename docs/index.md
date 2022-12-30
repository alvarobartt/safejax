# 🔐 Serialize JAX, Flax, Haiku, or Objax model params with `safetensors`

`safejax` is a Python package to serialize JAX, Flax, Haiku, or Objax model params using `safetensors`
as the tensor storage format, instead of relying on `pickle`. For more details on why
`safetensors` is safer than `pickle` please check [huggingface/safetensors](https://github.com/huggingface/safetensors).

Note that `safejax` supports the serialization of `jax`, `flax`, `dm-haiku`, and `objax` model
parameters and has been tested with all those frameworks, but there may be some cases where it
does not work as expected, as this is still in an early development phase, so please if you have
any feedback or bug reports, open an issue at [safejax/issues](https://github.com/alvarobartt/safejax/issues).
