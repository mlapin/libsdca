"""Deterministic JAX rematerialization and batch-sharded direct loss pullbacks."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from libsdca.jax import distillation_loss


def test_remat_and_batch_sharded_distillation() -> None:
    rng = np.random.default_rng(193)
    z = jnp.asarray(rng.normal(size=(8, 4097)), jnp.bfloat16)
    teacher = jnp.asarray(rng.normal(size=z.shape), jnp.bfloat16)
    factors = jnp.asarray([1.0, -1.25, 0.0, 2.0, 0.5, -1.0, 3.0, -0.25], jnp.float32)

    def loss(x: jax.Array, t: jax.Array) -> jax.Array:
        return distillation_loss(x, t, temperature=2, reduction="none", backend="kernel")

    def evaluate(
        fn: Callable[[jax.Array, jax.Array], jax.Array], x: jax.Array, t: jax.Array, u: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        values, pull = jax.vjp(lambda a: fn(a, t), x)
        return values, pull(u)[0]

    direct = jax.jit(lambda x, t, u: evaluate(loss, x, t, u))(z, teacher, factors)
    remat = jax.checkpoint(loss)
    mapped = jax.jit(lambda x, t, u: evaluate(remat, x, t, u))(z, teacher, factors)
    for a, b in zip(direct, mapped, strict=True):
        np.testing.assert_array_equal(a, b)

    mesh = jax.sharding.Mesh(np.asarray(jax.local_devices()[:2]), ("batch",))
    spec = jax.sharding.PartitionSpec("batch", None)
    score_sharding = jax.sharding.NamedSharding(mesh, spec)
    row_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("batch"))
    placed = jax.jit(
        lambda x, t, u: evaluate(remat, x, t, u),
        in_shardings=(score_sharding, score_sharding, row_sharding),
        out_shardings=(row_sharding, score_sharding),
    )(z, teacher, factors)
    for a, b in zip(direct, placed, strict=True):
        np.testing.assert_array_equal(a, b)
    if hasattr(jax, "shard_map"):
        sharded = jax.shard_map(
            remat, mesh=mesh, in_specs=(spec, spec), out_specs=jax.sharding.PartitionSpec("batch")
        )
    else:
        from jax.experimental.shard_map import shard_map

        # Older JAX has no replication rule for its native-library callback.
        # Independently compare the actual sharded outputs below.
        sharded = shard_map(
            remat,
            mesh=mesh,
            in_specs=(spec, spec),
            out_specs=jax.sharding.PartitionSpec("batch"),
            check_rep=False,
        )
    out = jax.jit(lambda x, t, u: evaluate(sharded, x, t, u))(z, teacher, factors)
    for a, b in zip(direct, out, strict=True):
        np.testing.assert_array_equal(a, b)
