import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict

from safejax.typing import PathLike


class SingleLayerModel(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.features)(x)
        return x


@pytest.fixture
def single_layer_model() -> nn.Module:
    return SingleLayerModel(features=1)


@pytest.fixture
def single_layer_model_params(single_layer_model: nn.Module) -> FrozenDict:
    # https://docs.pytest.org/en/7.1.x/how-to/fixtures.html#fixtures-can-request-other-fixtures
    rng = jax.random.PRNGKey(0)
    params = single_layer_model.init(rng, jnp.ones((1, 1)))
    return params


@pytest.fixture(scope="session")
def safetensors_file(tmp_path_factory) -> PathLike:
    # https://docs.pytest.org/en/7.1.x/how-to/tmp_path.html#the-tmp-path-factory-fixture
    return tmp_path_factory.mktemp("data") / "model.safetensors"
