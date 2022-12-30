from pathlib import Path

import pytest
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.serialization import (
    from_bytes,
    from_state_dict,
    msgpack_restore,
    msgpack_serialize,
    to_bytes,
    to_state_dict,
)

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
    assert isinstance(decoded_params, FrozenDict)
    assert len(decoded_params) > 0
    assert id(decoded_params) != id(params)
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
    assert isinstance(decoded_params, FrozenDict)
    assert len(decoded_params) > 0
    assert id(decoded_params) != id(params)
    assert decoded_params.keys() == params.keys()


@pytest.mark.parametrize(
    "params",
    [
        pytest.lazy_fixture("single_layer_params"),
        pytest.lazy_fixture("flax_resnet50_params"),
    ],
)
@pytest.mark.usefixtures("safetensors_file", "msgpack_file")
def test_safejax_and_msgpack(
    params: FlaxParams, safetensors_file: Path, msgpack_file: Path
) -> None:
    safetensors_file = serialize(params=params, filename=safetensors_file)
    assert isinstance(safetensors_file, Path)
    assert safetensors_file.exists()

    safetensors_decoded_params = deserialize(path_or_buf=safetensors_file)
    assert isinstance(safetensors_decoded_params, FrozenDict)
    assert len(safetensors_decoded_params) > 0
    assert id(safetensors_decoded_params) != id(params)
    assert safetensors_decoded_params.keys() == params.keys()

    with open(msgpack_file, mode="wb") as f:
        f.write(msgpack_serialize(unfreeze(params)))

    with open(msgpack_file, "rb") as f:
        msgpack_decoded_params = freeze(msgpack_restore(f.read()))

    assert isinstance(msgpack_decoded_params, type(params))
    assert len(msgpack_decoded_params) > 0
    assert id(msgpack_decoded_params) != id(params)
    assert msgpack_decoded_params.keys() == params.keys()

    params = flatten_dict(params)
    safetensors_decoded_params = flatten_dict(safetensors_decoded_params)
    msgpack_decoded_params = flatten_dict(msgpack_decoded_params)
    assert safetensors_decoded_params.keys() == msgpack_decoded_params.keys()
    assert all(
        safetensors_decoded_params[k].shape == msgpack_decoded_params[k].shape
        for k in params.keys()
    )


@pytest.mark.parametrize(
    "params",
    [
        pytest.lazy_fixture("single_layer_params"),
        pytest.lazy_fixture("flax_resnet50_params"),
    ],
)
def test_safejax_and_msgpack_bytes(params: FlaxParams) -> None:
    encoded_params = serialize(params=params)
    assert isinstance(encoded_params, bytes)
    assert len(encoded_params) > 0

    safetensors_decoded_params = deserialize(path_or_buf=encoded_params)
    assert isinstance(safetensors_decoded_params, FrozenDict)
    assert len(safetensors_decoded_params) > 0
    assert id(safetensors_decoded_params) != id(params)
    assert safetensors_decoded_params.keys() == params.keys()

    msgpack_bytes_encoded_params = to_bytes(params)
    assert isinstance(msgpack_bytes_encoded_params, bytes)
    assert len(msgpack_bytes_encoded_params) > 0

    msgpack_bytes_decoded_params = freeze(
        from_bytes(params, msgpack_bytes_encoded_params)
    )
    assert isinstance(msgpack_bytes_decoded_params, FrozenDict)
    assert len(msgpack_bytes_decoded_params) > 0
    assert id(msgpack_bytes_decoded_params) != id(params)
    assert msgpack_bytes_decoded_params.keys() == params.keys()

    params = flatten_dict(params)
    safetensors_decoded_params = flatten_dict(safetensors_decoded_params)
    msgpack_bytes_decoded_params = flatten_dict(msgpack_bytes_decoded_params)
    assert safetensors_decoded_params.keys() == msgpack_bytes_decoded_params.keys()
    assert all(
        safetensors_decoded_params[k].shape == msgpack_bytes_decoded_params[k].shape
        for k in params.keys()
    )


@pytest.mark.parametrize(
    "params",
    [
        pytest.lazy_fixture("single_layer_params"),
        # pytest.lazy_fixture("flax_resnet50_params"),
    ],
)
def test_safejax_and_state_dict(params: FlaxParams) -> None:
    encoded_params = serialize(params=params)
    assert isinstance(encoded_params, bytes)
    assert len(encoded_params) > 0

    safetensors_decoded_params = deserialize(path_or_buf=encoded_params)
    assert isinstance(safetensors_decoded_params, FrozenDict)
    assert len(safetensors_decoded_params) > 0
    assert id(safetensors_decoded_params) != id(params)
    assert safetensors_decoded_params.keys() == params.keys()

    state_dict_encoded_params = to_state_dict(params)
    assert isinstance(state_dict_encoded_params, dict)
    assert len(state_dict_encoded_params) > 0

    state_dict_decoded_params = from_state_dict(params, state_dict_encoded_params)
    assert isinstance(state_dict_decoded_params, FrozenDict)
    assert len(state_dict_decoded_params) > 0
    assert id(state_dict_decoded_params) != id(params)
    assert state_dict_decoded_params.keys() == params.keys()

    params = flatten_dict(params)
    safetensors_decoded_params = flatten_dict(safetensors_decoded_params)
    state_dict_decoded_params = flatten_dict(state_dict_decoded_params)
    assert safetensors_decoded_params.keys() == state_dict_decoded_params.keys()
    assert all(
        safetensors_decoded_params[k].shape == state_dict_decoded_params[k].shape
        for k in params.keys()
    )
