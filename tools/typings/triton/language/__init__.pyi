# Triton's JIT DSL does not follow Python typing rules; treat it as untyped.
from typing import Any

def __getattr__(name: str) -> Any: ...
