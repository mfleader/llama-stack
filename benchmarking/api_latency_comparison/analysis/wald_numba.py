# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Numba dispatch registration for PyMC's Wald (Inverse Gaussian) distribution.

PyMC's Wald (Inverse Gaussian) RV lacks a numba dispatch, so nutpie
compilation fails without this shim. Importing this module registers
the dispatch once.
"""

from typing import Any

from pymc.distributions.continuous import WaldRV as _PyMCWaldRV
from pytensor.link.numba.dispatch import basic as _numba_basic
from pytensor.link.numba.dispatch.random import numba_core_rv_funcify


@numba_core_rv_funcify.register(_PyMCWaldRV)
def _numba_wald_dispatch(op: Any, node: Any) -> Any:
    """Return a numba-jitted Wald (Inverse Gaussian) sampler for nutpie compilation."""

    @_numba_basic.numba_njit
    def random(rng, mu, lam, alpha):
        return rng.wald(mu, lam) + alpha

    return random
