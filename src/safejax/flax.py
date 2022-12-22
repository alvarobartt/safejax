from typing import Any, Dict, Union

import jax
import numpy as np
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict, freeze
from jax import numpy as jnp
from safetensors.flax import load, save


class SingleLayer(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=1)(x)
        return x


model = SingleLayer()
rng = jax.random.PRNGKey(0)
params = model.init(rng, jnp.ones((1, 1)))


def flatten_frozen_dict(
    frozen_or_unfrozen_dict: Union[Dict[str, Any], FrozenDict],
    key_prefix: Union[str, None] = None,
) -> Dict[str, jnp.DeviceArray]:
    """Idea from https://gist.github.com/Narsil/d5b0d747e5c8c299eb6d82709e480e3d"""
    weights = {}
    for key, value in frozen_or_unfrozen_dict.items():
        key = f"{key_prefix}.{key}" if key_prefix else key
        if isinstance(value, jnp.DeviceArray) or isinstance(value, np.ndarray):
            weights[key] = value
            continue
        if isinstance(value, FrozenDict) or isinstance(value, Dict):
            weights.update(
                flatten_frozen_dict(frozen_or_unfrozen_dict=value, key_prefix=key)
            )
    return weights


expected_format = {
    "params.Dense_0.kernel": jnp.DeviceArray,
    "params.Dense_0.bias": jnp.DeviceArray,
}

flattened_dict = flatten_frozen_dict(frozen_or_unfrozen_dict=params)

for key, value in expected_format.items():
    assert key in flattened_dict.keys()
    assert isinstance(flattened_dict[key], value)

saved_dict = save(tensors=flattened_dict)
loaded_dict = load(data=saved_dict)


def unflatten_frozen_dict(tensors: Dict[str, jnp.DeviceArray]) -> FrozenDict:
    """Idea from https://stackoverflow.com/a/63545677"""
    res = {}
    for k, v in tensors.items():
        res_tmp = res
        levels = k.split(".")
        for level in levels[:-1]:
            res_tmp = res_tmp.setdefault(level, {})
        res_tmp[levels[-1]] = v
    return freeze(res)


frozen_dict = unflatten_frozen_dict(tensors=loaded_dict)
y = model.apply(frozen_dict, jnp.ones((1, 1)))
