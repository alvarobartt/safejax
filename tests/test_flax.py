from pathlib import Path

import pytest
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.serialization import msgpack_restore, msgpack_serialize

from safejax.flax import deserialize, serialize
from safejax.typing import FlaxParams
from safejax.utils import flatten_dict


@pytest.mark.parametrize(
    "params",
    [
        pytest.lazy_fixture("single_layer_params"),
        pytest.lazy_fixture("flax_resnet50_params"),
    ],
)
def test_partial_deserialize(params: FlaxParams) -> None:
    encoded_params = serialize(params=params)
    decoded_params = deserialize(path_or_buf=encoded_params)
    assert len(decoded_params) > 0
    assert isinstance(decoded_params, FrozenDict)
    assert decoded_params.keys() == params.keys()


@pytest.mark.parametrize(
    "params",
    [
        pytest.lazy_fixture("single_layer_params"),
        pytest.lazy_fixture("flax_resnet50_params"),
    ],
)
@pytest.mark.usefixtures("safetensors_file")
def test_partial_deserialize_from_file(
    params: FlaxParams, safetensors_file: Path
) -> None:
    safetensors_file = serialize(params=params, filename=safetensors_file)
    decoded_params = deserialize(path_or_buf=safetensors_file)
    assert len(decoded_params) > 0
    assert isinstance(decoded_params, FrozenDict)
    assert decoded_params.keys() == params.keys()


@pytest.mark.parametrize(
    "params",
    [
        pytest.lazy_fixture("single_layer_params"),
        pytest.lazy_fixture("flax_resnet50_params"),
    ],
)
@pytest.mark.usefixtures("safetensors_file", "msgpack_file")
def test_safejax_versus_msgpack(
    params: FlaxParams, safetensors_file: Path, msgpack_file: Path
) -> None:
    safetensors_file = serialize(params=params, filename=safetensors_file)
    safetensors_decoded_params = deserialize(path_or_buf=safetensors_file)
    assert len(safetensors_decoded_params) > 0
    assert isinstance(safetensors_decoded_params, FrozenDict)
    assert safetensors_decoded_params.keys() == params.keys()

    with open(msgpack_file, mode="wb") as f:
        f.write(msgpack_serialize(unfreeze(params)))

    with open(msgpack_file, "rb") as f:
        msgpack_decoded_params = freeze(msgpack_restore(f.read()))

    assert len(msgpack_decoded_params) > 0
    assert isinstance(msgpack_decoded_params, type(params))
    assert msgpack_decoded_params.keys() == params.keys()

    params = flatten_dict(params)
    safetensors_decoded_params = flatten_dict(safetensors_decoded_params)
    msgpack_decoded_params = flatten_dict(msgpack_decoded_params)
    assert safetensors_decoded_params.keys() == msgpack_decoded_params.keys()
    assert all(
        safetensors_decoded_params[k].shape == msgpack_decoded_params[k].shape
        for k in params.keys()
    )
