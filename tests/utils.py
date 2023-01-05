import warnings

import chex
import jax
from flax.core.frozen_dict import FrozenDict, unfreeze
from objax.variable import VarCollection

from safejax.typing import ParamsDictLike


def assert_over_trees(params: ParamsDictLike, decoded_params: ParamsDictLike) -> None:
    """Assertions using `chex` to compare two trees of parameters.

    Note:
        This function does not support `objax.variable.VarCollection` objects yet,
        so the assertions are just done over `jax`, `flax`, and `haiku` params.

    Args:
        params: a `ParamsDictLike` object with the original parameters.
        decoded_params: a `ParamsDictLike` object with the decoded parameters using `safejax`.

    Raises:
        AssertionError: if the two trees are not equal on dtype, shape, structure, and values.
    """
    if isinstance(params, VarCollection) or isinstance(decoded_params, VarCollection):
        warnings.warn(
            "This function does not support `objax.variable.VarCollection` objects yet."
        )
    else:
        params = unfreeze(params) if isinstance(params, FrozenDict) else params
        decoded_params = (
            unfreeze(decoded_params)
            if isinstance(decoded_params, FrozenDict)
            else decoded_params
        )
        params_tree = jax.tree_util.tree_map(lambda x: x, params)
        decoded_params_tree = jax.tree_util.tree_map(lambda x: x, decoded_params)

        chex.assert_trees_all_close(
            params_tree, decoded_params_tree
        )  # static and jittable static
        chex.assert_trees_all_equal_dtypes(params_tree, decoded_params_tree)
        chex.assert_trees_all_equal_shapes(params_tree, decoded_params_tree)
        chex.assert_trees_all_equal_structs(params_tree, decoded_params_tree)
