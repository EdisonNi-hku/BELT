import torch
import transformers
from transformers.cache_utils import DynamicCache


def clone_cache(cache: DynamicCache) -> DynamicCache:
    """Move a DynamicCache to the specified device (cpu or cuda)."""
    new_cache = DynamicCache()
    # Handle empty cache
    if len(cache) == 0:
        return new_cache
    for layer_idx in range(len(cache)):
        if transformers.__version__.split('.')[0] == '5':
            key_states = cache.layers[layer_idx].keys
            value_states = cache.layers[layer_idx].values
        else:
            key_states, value_states = cache[layer_idx]
        new_cache.update(
            key_states.clone(),
            value_states.clone(),
            layer_idx,
        )
    return new_cache

def cache_to_device(cache: DynamicCache, device: str | torch.device) -> DynamicCache:
    """Move a DynamicCache to the specified device (cpu or cuda)."""
    new_cache = DynamicCache()
    # Handle empty cache
    if len(cache) == 0:
        return new_cache
    for layer_idx in range(len(cache)):
        if transformers.__version__.split('.')[0] == '5':
            key_states = cache.layers[layer_idx].keys
            value_states = cache.layers[layer_idx].values
        else:
            key_states, value_states = cache[layer_idx]
        new_cache.update(
            key_states.to(device),
            value_states.to(device),
            layer_idx,
        )
    return new_cache

def get_cache_devices(cache: DynamicCache) -> list[torch.device]:
    # new api for transformers v5.3
    return [cache.layers[layer_idx].keys.device for layer_idx in range(len(cache))]
