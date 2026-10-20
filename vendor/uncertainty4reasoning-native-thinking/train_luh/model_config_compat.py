"""Load a base model's config even when transformers cannot parse its model_type.

The vLLM-backed trainer runs in a container pinned to ``transformers<5``, because
transformers main violates the vLLM and compressed-tensors pins. Qwen3.5 support
there comes from vLLM's own model registry, not from transformers -- so
``AutoConfig.from_pretrained`` raises ``ValueError: ... model type `qwen3_5` ...``
even though vLLM serves the model perfectly well. Verified on Clariden
(job 2910637): AutoConfig fails both before and after importing vllm, while
``vllm.transformers_utils.config.get_config`` succeeds.

The trainer only needs two numbers off the config -- ``hidden_size`` to size the
uncertainty head's feature extractor, and ``num_hidden_layers`` to resolve
negative hidden-state layer indices -- so falling back to vLLM's loader is
sufficient.

The fallback fires only when AutoConfig raises, so behaviour is unchanged for
every model transformers already understands.
"""

import logging

log = logging.getLogger(__name__)

# Multimodal wrappers (Qwen3.5 among them) nest the language model's dimensions
# under a sub-config, so the fields are not always at the top level.
_NESTED_CONFIG_ATTRS = ("text_config", "llm_config", "language_config", "decoder")
_REQUIRED_FIELDS = ("hidden_size", "num_hidden_layers")


def _find_field(config, field: str, depth: int = 0):
    """Look for ``field`` on ``config``, then in known nested sub-configs."""
    if config is None or depth > 3:
        return None
    value = getattr(config, field, None)
    if isinstance(value, int):
        return value
    for attr in _NESTED_CONFIG_ATTRS:
        found = _find_field(getattr(config, attr, None), field, depth + 1)
        if found is not None:
            return found
    return None


class ResolvedBaseConfig:
    """Exposes hidden_size / num_hidden_layers, delegating everything else.

    Delegation matters: callers may read other attributes off the config, and
    silently dropping them would turn a clear AttributeError into a subtle
    misconfiguration.
    """

    def __init__(self, wrapped, hidden_size: int, num_hidden_layers: int):
        self._wrapped = wrapped
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

    def __getattr__(self, item):
        return getattr(self._wrapped, item)

    def __repr__(self):
        return (
            f"ResolvedBaseConfig(hidden_size={self.hidden_size}, "
            f"num_hidden_layers={self.num_hidden_layers}, "
            f"wrapped={type(self._wrapped).__name__})"
        )


def load_base_config(
    model_path: str,
    trust_remote_code: bool = True,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | None = None,
):
    """Return a config exposing hidden_size and num_hidden_layers.

    Tries transformers first, then vLLM's config loader. Raises if neither can
    produce both fields, rather than returning a config that would silently size
    the head wrongly.
    """
    from transformers import AutoConfig

    config = None
    transformers_kwargs = {"trust_remote_code": trust_remote_code}
    for key, value in (
        ("revision", revision),
        ("cache_dir", cache_dir),
        ("token", token),
    ):
        if value is not None:
            transformers_kwargs[key] = value
    try:
        config = AutoConfig.from_pretrained(model_path, **transformers_kwargs)
    except Exception as transformers_error:  # noqa: BLE001 - fall back, then re-report
        log.warning(
            "transformers AutoConfig could not load %s (%s); falling back to vLLM's "
            "config loader",
            model_path,
            transformers_error,
        )
        try:
            from vllm.transformers_utils.config import get_config

            vllm_kwargs = {"trust_remote_code": trust_remote_code}
            if revision is not None:
                vllm_kwargs["revision"] = revision
            config = get_config(model_path, **vllm_kwargs)
        except Exception as vllm_error:  # noqa: BLE001
            raise RuntimeError(
                f"neither transformers nor vLLM could load a config for {model_path}: "
                f"transformers said {transformers_error!r}; vLLM said {vllm_error!r}"
            ) from vllm_error

    resolved = {field: _find_field(config, field) for field in _REQUIRED_FIELDS}
    missing = [field for field, value in resolved.items() if value is None]
    if missing:
        raise RuntimeError(
            f"config for {model_path} ({type(config).__name__}) is missing {missing}; "
            "the uncertainty head cannot be sized without them"
        )
    return ResolvedBaseConfig(config, resolved["hidden_size"], resolved["num_hidden_layers"])
