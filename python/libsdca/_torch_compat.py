"""PyTorch version differences used by the tensor front ends."""

import torch

if hasattr(torch.compiler, "is_compiling"):
    from torch.compiler import is_compiling as is_compiling
else:  # PyTorch 2.1 and 2.2.
    from torch._dynamo import is_compiling as is_compiling
