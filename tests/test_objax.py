from pathlib import Path

import pytest
from jax import numpy as jnp
from objax.variable import VarCollection

from safejax.objax import deserialize, deserialize_with_assignment, serialize
from safejax.typing import ObjaxParams


@pytest.mark.parametrize(
    "params",
    [
        pytest.lazy_fixture("objax_single_layer_params"),
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
    assert id(decoded_params) != id(params)
    assert decoded_params.keys() == params.keys()


@pytest.mark.parametrize(
    "params",
    [
        pytest.lazy_fixture("objax_single_layer_params"),
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
    assert id(decoded_params) != id(params)
    assert decoded_params.keys() == params.keys()


@pytest.mark.parametrize(
    "params",
    [
        pytest.lazy_fixture("objax_single_layer_params"),
        pytest.lazy_fixture("objax_resnet50_params"),
    ],
)
@pytest.mark.usefixtures("safetensors_file")
def test_serialize_and_deserialize_with_assignment(
    params: ObjaxParams, safetensors_file: Path
) -> None:
    safetensors_file = serialize(params=params, filename=safetensors_file)
    assert isinstance(safetensors_file, Path)
    assert safetensors_file.exists()

    # Assign jnp.zeros to all params.tensors() to make sure the assignment is working
    # before we deserialize the params.
    params.assign([jnp.zeros(x.shape, x.dtype) for x in params.tensors()])
    assert all([jnp.all(x == 0) for x in params.tensors()])

    deserialize_with_assignment(filename=safetensors_file, model_vars=params)
    assert isinstance(params, VarCollection)
    assert len(params) > 0
    assert not all([jnp.all(x != 0) for x in params.tensors()])
