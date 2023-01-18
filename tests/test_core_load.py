from pathlib import Path
from typing import Any, Dict, Union

import pytest
from flax.core.frozen_dict import FrozenDict
from fsspec.spec import AbstractFileSystem
from objax.variable import VarCollection

from safejax.core.load import deserialize
from safejax.core.save import serialize
from safejax.typing import ParamsDictLike

from .utils import assert_over_trees


@pytest.mark.parametrize(
    "params, serialize_kwargs, deserialize_kwargs, expected_output_type",
    [
        (
            pytest.lazy_fixture("flax_resnet50_params"),
            {},
            {"freeze_dict": True},
            FrozenDict,
        ),
        (pytest.lazy_fixture("flax_resnet50_params"), {}, {"freeze_dict": False}, dict),
        (
            pytest.lazy_fixture("objax_resnet50_params"),
            {"include_objax_variables": True},
            {"requires_unflattening": False, "to_var_collection": True},
            VarCollection,
        ),
        (
            pytest.lazy_fixture("objax_resnet50_params"),
            {"include_objax_variables": False},
            {"requires_unflattening": False, "to_var_collection": False},
            dict,
        ),
        (pytest.lazy_fixture("haiku_resnet50_params"), {}, {}, dict),
    ],
)
def test_deserialize(
    params: ParamsDictLike,
    serialize_kwargs: Dict[str, Any],
    deserialize_kwargs: Dict[str, Any],
    expected_output_type: Union[dict, FrozenDict, VarCollection],
) -> None:
    encoded_params = serialize(params=params, **serialize_kwargs)
    decoded_params = deserialize(path_or_buf=encoded_params, **deserialize_kwargs)
    assert isinstance(decoded_params, expected_output_type)
    assert len(decoded_params) > 0
    assert id(decoded_params) != id(params)
    assert decoded_params.keys() == params.keys()

    assert_over_trees(params=params, decoded_params=decoded_params)


@pytest.mark.parametrize(
    "params, serialize_kwargs, deserialize_kwargs, expected_output_type",
    [
        (
            pytest.lazy_fixture("flax_resnet50_params"),
            {},
            {"freeze_dict": True},
            FrozenDict,
        ),
        (pytest.lazy_fixture("flax_resnet50_params"), {}, {"freeze_dict": False}, dict),
        (
            pytest.lazy_fixture("objax_resnet50_params"),
            {"include_objax_variables": True},
            {"requires_unflattening": False, "to_var_collection": True},
            VarCollection,
        ),
        (
            pytest.lazy_fixture("objax_resnet50_params"),
            {"include_objax_variables": False},
            {"requires_unflattening": False, "to_var_collection": False},
            dict,
        ),
        (pytest.lazy_fixture("haiku_resnet50_params"), {}, {}, dict),
    ],
)
@pytest.mark.usefixtures("safetensors_file")
def test_deserialize_from_file(
    params: ParamsDictLike,
    serialize_kwargs: Dict[str, Any],
    deserialize_kwargs: Dict[str, Any],
    expected_output_type: Union[dict, FrozenDict, VarCollection],
    safetensors_file: Path,
) -> None:
    safetensors_file = serialize(
        params=params, filename=safetensors_file, **serialize_kwargs
    )
    decoded_params = deserialize(path_or_buf=safetensors_file, **deserialize_kwargs)
    assert isinstance(decoded_params, expected_output_type)
    assert len(decoded_params) > 0
    assert id(decoded_params) != id(params)
    assert decoded_params.keys() == params.keys()

    assert_over_trees(params=params, decoded_params=decoded_params)


@pytest.mark.usefixtures("flax_resnet50_params", "safetensors_file", "metadata")
def test_deserialize_from_file_with_metadata(
    flax_resnet50_params: ParamsDictLike,
    safetensors_file: Path,
    metadata: Dict[str, str],
) -> None:
    safetensors_file = serialize(
        params=flax_resnet50_params, filename=safetensors_file, metadata=metadata
    )
    decoded_params, metadata = deserialize(path_or_buf=safetensors_file)
    assert isinstance(decoded_params, dict)
    assert len(decoded_params) > 0
    assert id(decoded_params) != id(flax_resnet50_params)
    assert decoded_params.keys() == flax_resnet50_params.keys()
    assert metadata is not None
    assert isinstance(metadata, dict)
    assert len(metadata) > 0

    assert_over_trees(params=flax_resnet50_params, decoded_params=decoded_params)


@pytest.mark.parametrize(
    "params, serialize_kwargs, deserialize_kwargs, expected_output_type",
    [
        (
            pytest.lazy_fixture("flax_resnet50_params"),
            {},
            {"freeze_dict": True},
            FrozenDict,
        ),
        (pytest.lazy_fixture("flax_resnet50_params"), {}, {"freeze_dict": False}, dict),
        (
            pytest.lazy_fixture("objax_resnet50_params"),
            {"include_objax_variables": True},
            {"requires_unflattening": False, "to_var_collection": True},
            VarCollection,
        ),
        (
            pytest.lazy_fixture("objax_resnet50_params"),
            {"include_objax_variables": False},
            {"requires_unflattening": False, "to_var_collection": False},
            dict,
        ),
        (pytest.lazy_fixture("haiku_resnet50_params"), {}, {}, dict),
    ],
)
@pytest.mark.usefixtures("safetensors_file", "fs")
def test_deserialize_from_file_in_fs(
    params: ParamsDictLike,
    serialize_kwargs: Dict[str, Any],
    deserialize_kwargs: Dict[str, Any],
    expected_output_type: Union[dict, FrozenDict, VarCollection],
    safetensors_file: Path,
    fs: AbstractFileSystem,
) -> None:
    safetensors_file = serialize(
        params=params, filename=safetensors_file, fs=fs, **serialize_kwargs
    )
    decoded_params = deserialize(
        path_or_buf=safetensors_file, fs=fs, **deserialize_kwargs
    )
    assert isinstance(decoded_params, expected_output_type)
    assert len(decoded_params) > 0
    assert id(decoded_params) != id(params)
    assert decoded_params.keys() == params.keys()

    assert_over_trees(params=params, decoded_params=decoded_params)
