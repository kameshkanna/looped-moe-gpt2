"""Cross-cutting utilities: device resolution and reproducibility helpers."""

from looped_moe_gpt2.utils.device import resolve_amp_dtype, resolve_device
from looped_moe_gpt2.utils.seed import set_seed

__all__ = ["resolve_device", "resolve_amp_dtype", "set_seed"]
