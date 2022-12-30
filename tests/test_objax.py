from pathlib import Path

import pytest
from objax.variable import VarCollection

from safejax.objax import deserialize, serialize
from safejax.typing import ObjaxParams


@pytest.mark.parametrize(
    "params",
    [
        pytest.lazy_fixture("objax_resnet50_params"),
    ],
)
def test_serialize_and_deserialize(params: ObjaxParams) -> None:
    encoded_params = serialize(params=params)
    assert isinstance(encoded_params, bytes)
    assert len(encoded_params) > 0

    decoded_params = deserialize(path_or_buf=encoded_params)
    assert isinstance(decoded_params, VarCollection)
    assert len(decoded_params) > 0
    assert decoded_params.keys() == params.keys()


@pytest.mark.parametrize(
    "params",
    [
        pytest.lazy_fixture("objax_resnet50_params"),
    ],
)
@pytest.mark.usefixtures("safetensors_file")
def test_serialize_and_deserialize_from_file(
    params: ObjaxParams, safetensors_file: Path
) -> None:
    safetensors_file = serialize(params=params, filename=safetensors_file)
    assert isinstance(safetensors_file, Path)
    assert safetensors_file.exists()

    decoded_params = deserialize(path_or_buf=safetensors_file)
    assert isinstance(decoded_params, VarCollection)
    assert len(decoded_params) > 0
    assert decoded_params.keys() == params.keys()
