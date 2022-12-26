from typing import Any, Dict

import pytest
from jax import numpy as jnp

from safejax.utils import flatten_dict, unflatten_dict


@pytest.mark.parametrize(
    "input_dict, expected_output_dict",
    [
        (
            {"a": jnp.zeros(1), "b": jnp.zeros(1)},
            {"a": jnp.zeros(1), "b": jnp.zeros(1)},
        ),
        (
            {"a.b": jnp.zeros(1), "b": jnp.zeros(1)},
            {"a": {"b": jnp.zeros(1)}, "b": jnp.zeros(1)},
        ),
        (
            {"a.b": jnp.zeros(1), "a.c": jnp.zeros(1), "b": jnp.zeros(1)},
            {"a": {"b": jnp.zeros(1), "c": jnp.zeros(1)}, "b": jnp.zeros(1)},
        ),
        (
            {
                "a.b.c": jnp.zeros(1),
                "a.b.d": jnp.zeros(1),
                "a.e": jnp.zeros(1),
                "b": jnp.zeros(1),
            },
            {
                "a": {"b": {"c": jnp.zeros(1), "d": jnp.zeros(1)}, "e": jnp.zeros(1)},
                "b": jnp.zeros(1),
            },
        ),
        (
            {
                "a.b.c": jnp.zeros(1),
                "a.b.d": jnp.zeros(1),
                "a.e": jnp.zeros(1),
                "b": jnp.zeros(1),
                "c": jnp.zeros(1),
            },
            {
                "a": {"b": {"c": jnp.zeros(1), "d": jnp.zeros(1)}, "e": jnp.zeros(1)},
                "b": jnp.zeros(1),
                "c": jnp.zeros(1),
            },
        ),
    ],
)
def test_unflatten_dict(
    input_dict: Dict[str, Any], expected_output_dict: Dict[str, Any]
) -> None:
    unflattened_dict = unflatten_dict(tensors=input_dict)
    assert unflattened_dict == expected_output_dict


@pytest.mark.parametrize(
    "input_dict, expected_output_dict",
    [
        (
            {"a": {"b": jnp.zeros(1)}, "b": jnp.zeros(1)},
            {"a.b": jnp.zeros(1), "b": jnp.zeros(1)},
        ),
        (
            {"a": {"b": jnp.zeros(1), "c": jnp.zeros(1)}, "b": jnp.zeros(1)},
            {"a.b": jnp.zeros(1), "a.c": jnp.zeros(1), "b": jnp.zeros(1)},
        ),
        (
            {
                "a": {"b": {"c": jnp.zeros(1), "d": jnp.zeros(1)}, "e": jnp.zeros(1)},
                "b": jnp.zeros(1),
            },
            {
                "a.b.c": jnp.zeros(1),
                "a.b.d": jnp.zeros(1),
                "a.e": jnp.zeros(1),
                "b": jnp.zeros(1),
            },
        ),
        (
            {
                "a": {"b": {"c": jnp.zeros(1), "d": jnp.zeros(1)}, "e": jnp.zeros(1)},
                "b": jnp.zeros(1),
                "c": jnp.zeros(1),
            },
            {
                "a.b.c": jnp.zeros(1),
                "a.b.d": jnp.zeros(1),
                "a.e": jnp.zeros(1),
                "b": jnp.zeros(1),
                "c": jnp.zeros(1),
            },
        ),
    ],
)
def test_flatten_dict(
    input_dict: Dict[str, Any], expected_output_dict: Dict[str, Any]
) -> None:
    flattened_dict = flatten_dict(params=input_dict)
    assert flattened_dict == expected_output_dict
