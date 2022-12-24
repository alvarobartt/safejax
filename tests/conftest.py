from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from flaxmodels.resnet import ResNet50


class SingleLayer(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.features)(x)
        return x


@pytest.fixture
def single_layer() -> nn.Module:
    return SingleLayer(features=1)


@pytest.fixture
def single_layer_params(single_layer: nn.Module) -> FrozenDict:
    # https://docs.pytest.org/en/7.1.x/how-to/fixtures.html#fixtures-can-request-other-fixtures
    rng = jax.random.PRNGKey(0)
    params = single_layer.init(rng, jnp.ones((1, 1)))
    return params


@pytest.fixture
def resnet50() -> nn.Module:
    return ResNet50()


@pytest.fixture
def resnet50_params(resnet50: nn.Module) -> FrozenDict:
    rng = jax.random.PRNGKey(0)
    params = resnet50.init(rng, jnp.ones((1, 224, 224, 3)))
    return params


@pytest.fixture(scope="session")
def safetensors_file(tmp_path_factory) -> Path:
    # https://docs.pytest.org/en/7.1.x/how-to/tmp_path.html#the-tmp-path-factory-fixture
    return Path(tmp_path_factory.mktemp("data") / "model.safetensors")
