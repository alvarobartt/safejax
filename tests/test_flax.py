from pathlib import Path

import pytest
from flax.core.frozen_dict import FrozenDict

from safejax import deserialize, serialize


@pytest.mark.parametrize(
    "params",
    [
        pytest.lazy_fixture("single_layer_params"),
        pytest.lazy_fixture("resnet50_params"),
    ],
)
def test_serialize(params: FrozenDict) -> None:
    serialized = serialize(params=params)
    assert isinstance(serialized, bytes)
    assert len(serialized) > 0


@pytest.mark.parametrize(
    "params",
    [
        pytest.lazy_fixture("single_layer_params"),
        pytest.lazy_fixture("resnet50_params"),
    ],
)
@pytest.mark.usefixtures("safetensors_file")
def test_serialize_to_file(params: FrozenDict, safetensors_file: Path) -> None:
    safetensors_file = serialize(params=params, filename=safetensors_file)
    assert isinstance(safetensors_file, Path)
    assert safetensors_file.exists()


@pytest.mark.parametrize(
    "params",
    [
        pytest.lazy_fixture("single_layer_params"),
        pytest.lazy_fixture("resnet50_params"),
    ],
)
def test_deserialize(params: FrozenDict) -> None:
    serialized = serialize(params=params)
    deserialized = deserialize(path_or_buf=serialized)
    assert isinstance(deserialized, FrozenDict)
    assert len(deserialized) > 0


@pytest.mark.parametrize(
    "params",
    [
        pytest.lazy_fixture("single_layer_params"),
        pytest.lazy_fixture("resnet50_params"),
    ],
)
@pytest.mark.usefixtures("safetensors_file")
def test_deserialize_from_file(params: FrozenDict, safetensors_file: Path) -> None:
    safetensors_file = serialize(params=params, filename=safetensors_file)
    deserialized = deserialize(path_or_buf=safetensors_file)
    assert isinstance(deserialized, FrozenDict)
    assert len(deserialized) > 0
