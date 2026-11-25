"""The CUDA kernels of libsdca in Triton: one module per loss family (as include/sdca/loss/),
shared device math and launch plans, and one launch layer per framework (launch_torch,
launch_jax). Importing this package imports neither framework."""

# The register-row kernels hold both distributions in one tile up to this capacity.
DISTILLATION_REGISTER_CLASSES = 4096
