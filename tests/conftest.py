from pathlib import Path
from typing import Dict

import fsspec
import haiku as hk
import jax
import jax.numpy as jnp
import objax
import pytest
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from flaxmodels.resnet import ResNet50 as FlaxResNet50
from fsspec.spec import AbstractFileSystem
from objax.variable import VarCollection
from objax.zoo.resnet_v2 import ResNet50 as ObjaxResNet50


@pytest.fixture
def flax_single_layer() -> nn.Module:
    class SingleLayer(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(features=1)(x)
            return x

    return SingleLayer()


@pytest.fixture
def flax_single_layer_params(flax_single_layer: nn.Module) -> FrozenDict:
    # https://docs.pytest.org/en/7.1.x/how-to/fixtures.html#fixtures-can-request-other-fixtures
    rng = jax.random.PRNGKey(0)
    params = flax_single_layer.init(rng, jnp.ones((1, 1)))
    return params


@pytest.fixture
def objax_single_layer() -> objax.nn.Sequential:
    return objax.nn.Sequential(
        [
            objax.nn.Linear(1, 1),
        ]
    )


@pytest.fixture
def objax_single_layer_params(objax_single_layer: objax.nn.Sequential) -> VarCollection:
    # https://docs.pytest.org/en/7.1.x/how-to/fixtures.html#fixtures-can-request-other-fixtures
    return objax_single_layer.vars()


@pytest.fixture
def flax_resnet() -> nn.Module:
    return FlaxResNet50()


@pytest.fixture
def flax_resnet50_params(flax_resnet: nn.Module) -> FrozenDict:
    rng = jax.random.PRNGKey(0)
    params = flax_resnet.init(rng, jnp.ones((1, 224, 224, 3)))
    return params


@pytest.fixture
def haiku_resnet50() -> hk.TransformedWithState:
    def resnet_fn(x: jnp.DeviceArray, is_training: bool) -> hk.Module:
        resnet = hk.nets.ResNet50(num_classes=10)
        return resnet(x, is_training=is_training)

    return hk.without_apply_rng(hk.transform_with_state(resnet_fn))


@pytest.fixture
def haiku_resnet50_params(haiku_resnet50: hk.TransformedWithState) -> FrozenDict:
    rng = jax.random.PRNGKey(0)
    params, _ = haiku_resnet50.init(rng, jnp.ones((1, 224, 224, 3)), is_training=True)
    return params


@pytest.fixture
def objax_resnet50() -> objax.nn.Sequential:
    return ObjaxResNet50(in_channels=3, num_classes=1000)


@pytest.fixture
def objax_resnet50_params(objax_resnet50: objax.nn.Sequential) -> VarCollection:
    return objax_resnet50.vars()


@pytest.fixture
def metadata() -> Dict[str, str]:
    return {
        "test": "test",
    }


@pytest.fixture(scope="session")
def safetensors_file(tmp_path_factory) -> Path:
    # https://docs.pytest.org/en/7.1.x/how-to/tmp_path.html#the-tmp-path-factory-fixture
    return Path(tmp_path_factory.mktemp("data") / "params.safetensors")


@pytest.fixture(scope="session")
def msgpack_file(tmp_path_factory) -> Path:
    return Path(tmp_path_factory.mktemp("data") / "params.msgpack")


@pytest.fixture
def fs() -> AbstractFileSystem:
    return fsspec.filesystem("file")
