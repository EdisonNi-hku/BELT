"""Public surface of the uncertainty-head package.

``AutoUncertaintyHead`` is imported eagerly because it depends only on this
package's own modules. The other three exports each pull in ``lm_polygraph``,
which is only needed for inference/estimation, so they are resolved lazily on
first attribute access (PEP 562).

This matters because ``lm_polygraph`` cannot be imported under transformers 5.x:
``lm_polygraph.utils.ensemble_utils.ensemble_sample`` imports ``SampleOutput``
from ``transformers.generation.utils``, which 5.x removed. Training a UHead on
Qwen3.5 requires transformers >= 5.14 (earlier versions ship no ``qwen3_5``
architecture), so an eager import here made the head untrainable on that model
for reasons unrelated to the head itself.

The public API is unchanged -- ``from luh import LuhClaimEstimator`` still works.
The only difference is that an ``lm_polygraph`` import error now surfaces when
that name is actually used, rather than when the package is imported.
"""

from .auto_uncertainty_head import AutoUncertaintyHead

__all__ = [
    "AutoUncertaintyHead",
    "CalculatorInferLuh",
    "CausalLMWithUncertainty",
    "LuhClaimEstimator",
]

# name -> submodule providing it; each of these imports lm_polygraph.
_LAZY_EXPORTS = {
    "LuhClaimEstimator": ".luh_claim_estimator",
    "CalculatorInferLuh": ".calculator_infer_luh",
    "CausalLMWithUncertainty": ".causal_lm_with_uncertainty",
}


def __getattr__(name):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value  # cache, so later accesses skip this path
    return value


def __dir__():
    return sorted(__all__)
