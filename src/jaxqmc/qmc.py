from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp


@jax.jit
@partial(jax.vmap, in_axes=0)
def vdc_base2(i: int) -> float:
    def body_fun(carry):
        i, inv_pow2, acc = carry
        bit = i & jnp.uint32(1)
        acc = acc + jnp.float32(bit) * inv_pow2
        i = i >> jnp.uint32(1)
        inv_pow2 = inv_pow2 * 0.5
        return (i, inv_pow2, acc)

    def cond_fun(carry):
        i, *_ = carry
        return i > 0

    # Start with 1/2 for the first reflected bit
    init = (i, jnp.float32(0.5), jnp.float32(0.0))
    _, _, acc = lax.while_loop(cond_fun, body_fun, init)
    return acc


def sobol_normal_1d(n: int, start_index: int = 1, dtype=jnp.float32) -> jnp.ndarray:
    idx = jnp.arange(start_index, start_index + n, dtype=jnp.uint32)
    u = vdc_base2(idx)  # in [0,1)
    # Map to standard normal: z = sqrt(2) * erfinv(2u - 1)
    z = jnp.sqrt(2.0) * lax.erf_inv((2.0 * u - 1.0).astype(dtype))
    z = jax.random.permutation(key=jax.random.PRNGKey(42), x=z)
    return z.astype(dtype)


def sobol_uniform_1d(n: int, start_index: int = 1, dtype=jnp.float32) -> jnp.ndarray:
    idx = jnp.arange(start_index, start_index + n, dtype=jnp.uint32)
    u = vdc_base2(idx)  # in [0,1)
    return u.astype(dtype)
