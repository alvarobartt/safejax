from pathlib import Path

import pytest
from flax.core.frozen_dict import FrozenDict

from safejax import deserialize, serialize


@pytest.mark.parametrize(
    "params",
    [
        pytest.lazy_fixture("single_layer_params"),
        pytest.lazy_fixture("flax_resnet50_params"),
    ],
)
def test_deserialize(params: FrozenDict) -> None:
    encoded_params = serialize(params=params)
    decoded_params = deserialize(path_or_buf=encoded_params, freeze_dict=True)
    assert isinstance(decoded_params, FrozenDict)
    assert len(decoded_params) > 0


@pytest.mark.parametrize(
    "params",
    [
        pytest.lazy_fixture("single_layer_params"),
        pytest.lazy_fixture("flax_resnet50_params"),
    ],
)
@pytest.mark.usefixtures("safetensors_file")
def test_deserialize_from_file(params: FrozenDict, safetensors_file: Path) -> None:
    safetensors_file = serialize(params=params, filename=safetensors_file)
    decoded_params = deserialize(path_or_buf=safetensors_file, freeze_dict=True)
    assert isinstance(decoded_params, FrozenDict)
    assert len(decoded_params) > 0
