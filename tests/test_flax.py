from pathlib import Path

import pytest
from flax.core.frozen_dict import FrozenDict

from safejax.flax import deserialize, serialize


@pytest.mark.usefixtures("single_layer_model_params")
def test_serialize(single_layer_model_params: FrozenDict) -> None:
    serialized = serialize(params=single_layer_model_params)
    assert isinstance(serialized, bytes)
    assert len(serialized) > 0


@pytest.mark.usefixtures("single_layer_model_params", "safetensors_file")
def test_serialize_to_file(
    single_layer_model_params: FrozenDict, safetensors_file: Path
) -> None:
    safetensors_file = serialize(
        params=single_layer_model_params, filename=safetensors_file
    )
    assert isinstance(safetensors_file, Path)
    assert safetensors_file.exists()


@pytest.mark.usefixtures("single_layer_model_params")
def test_deserialize(single_layer_model_params: FrozenDict) -> None:
    serialized = serialize(params=single_layer_model_params)
    deserialized = deserialize(path_or_buf=serialized)
    assert isinstance(deserialized, FrozenDict)
    assert len(deserialized) > 0


@pytest.mark.usefixtures("single_layer_model_params", "safetensors_file")
def test_deserialize_from_file(
    single_layer_model_params: FrozenDict, safetensors_file: Path
) -> None:
    safetensors_file = serialize(
        params=single_layer_model_params, filename=safetensors_file
    )
    deserialized = deserialize(path_or_buf=safetensors_file)
    assert isinstance(deserialized, FrozenDict)
    assert len(deserialized) > 0
