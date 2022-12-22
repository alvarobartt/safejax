import pytest

from flax.core.frozen_dict import FrozenDict

from safejax.flax import serialize, deserialize


@pytest.mark.usefixtures("single_layer_model", "single_layer_model_params")
def test_serialize(single_layer_model_params: FrozenDict) -> None:
    serialized = serialize(frozen_or_unfrozen_dict=single_layer_model_params)
    assert isinstance(serialized, bytes)
    assert len(serialized) > 0


@pytest.mark.usefixtures("single_layer_model", "single_layer_model_params")
def test_deserialize(single_layer_model_params: FrozenDict) -> None:
    serialized = serialize(frozen_or_unfrozen_dict=single_layer_model_params)
    deserialized = deserialize(data=serialized)
    assert isinstance(deserialized, FrozenDict)
    assert len(deserialized) > 0
