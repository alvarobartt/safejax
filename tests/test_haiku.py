from pathlib import Path

import pytest

from safejax.haiku import deserialize, serialize
from safejax.typing import HaikuParams

from .utils import assert_over_trees


@pytest.mark.parametrize(
    "params",
    [
        pytest.lazy_fixture("haiku_resnet50_params"),
    ],
)
def test_serialize_and_deserialize(params: HaikuParams) -> None:
    encoded_params = serialize(params=params)
    assert isinstance(encoded_params, bytes)
    assert len(encoded_params) > 0

    decoded_params = deserialize(path_or_buf=encoded_params)
    assert isinstance(decoded_params, dict)
    assert len(decoded_params) > 0
    assert id(decoded_params) != id(params)
    assert decoded_params.keys() == params.keys()

    assert_over_trees(params=params, decoded_params=decoded_params)


@pytest.mark.parametrize(
    "params",
    [
        pytest.lazy_fixture("haiku_resnet50_params"),
    ],
)
@pytest.mark.usefixtures("safetensors_file")
def test_serialize_and_deserialize_from_file(
    params: HaikuParams, safetensors_file: Path
) -> None:
    safetensors_file = serialize(params=params, filename=safetensors_file)
    assert isinstance(safetensors_file, Path)
    assert safetensors_file.exists()

    decoded_params = deserialize(path_or_buf=safetensors_file)
    assert isinstance(decoded_params, dict)
    assert len(decoded_params) > 0
    assert id(decoded_params) != id(params)
    assert decoded_params.keys() == params.keys()

    assert_over_trees(params=params, decoded_params=decoded_params)
