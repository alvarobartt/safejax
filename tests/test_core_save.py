from pathlib import Path

import pytest
from fsspec.spec import AbstractFileSystem

from safejax.core.save import serialize
from safejax.utils import ParamsDictLike


@pytest.mark.parametrize(
    "params",
    [
        pytest.lazy_fixture("flax_single_layer_params"),
        pytest.lazy_fixture("flax_resnet50_params"),
        pytest.lazy_fixture("objax_resnet50_params"),
        pytest.lazy_fixture("haiku_resnet50_params"),
    ],
)
def test_serialize(params: ParamsDictLike) -> None:
    encoded_params = serialize(params=params)
    assert isinstance(encoded_params, bytes)
    assert len(encoded_params) > 0


@pytest.mark.parametrize(
    "params",
    [
        pytest.lazy_fixture("flax_single_layer_params"),
        pytest.lazy_fixture("flax_resnet50_params"),
        pytest.lazy_fixture("objax_resnet50_params"),
        pytest.lazy_fixture("haiku_resnet50_params"),
    ],
)
@pytest.mark.usefixtures("safetensors_file")
def test_serialize_to_file(params: ParamsDictLike, safetensors_file: Path) -> None:
    safetensors_file = serialize(params=params, filename=safetensors_file)
    assert isinstance(safetensors_file, Path)
    assert safetensors_file.exists()


@pytest.mark.parametrize(
    "params",
    [
        pytest.lazy_fixture("flax_single_layer_params"),
        pytest.lazy_fixture("flax_resnet50_params"),
        pytest.lazy_fixture("objax_resnet50_params"),
        pytest.lazy_fixture("haiku_resnet50_params"),
    ],
)
@pytest.mark.usefixtures("safetensors_file", "fs")
def test_serialize_to_file_in_fs(
    params: ParamsDictLike, safetensors_file: Path, fs: AbstractFileSystem
) -> None:
    safetensors_file = serialize(params=params, filename=safetensors_file, fs=fs)
    assert isinstance(safetensors_file, Path)
    assert safetensors_file.exists()
    assert safetensors_file.as_posix() in fs.ls(safetensors_file.parent.as_posix())
