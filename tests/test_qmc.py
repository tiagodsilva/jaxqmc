import jax.numpy as jnp
import pytest  # noqa: F401

from jaxqmc import sobol_normal_1d, sobol_uniform_1d, vdc_base2


class TestVdcBase2:
    def test_output_shape(self):
        n = 10
        result = vdc_base2(jnp.arange(n, dtype=jnp.uint32))
        assert result.shape == (n,)

    def test_output_range(self):
        result = vdc_base2(jnp.arange(1, 11, dtype=jnp.uint32))
        assert jnp.all(result >= 0.0)
        assert jnp.all(result < 1.0)

    def test_deterministic(self):
        key1 = vdc_base2(jnp.arange(5, dtype=jnp.uint32))
        key2 = vdc_base2(jnp.arange(5, dtype=jnp.uint32))
        assert jnp.allclose(key1, key2)


class TestSobolUniform1d:
    def test_output_shape(self):
        n = 100
        result = sobol_uniform_1d(n)
        assert result.shape == (n,)

    def test_output_range(self):
        result = sobol_uniform_1d(100)
        assert jnp.all(result >= 0.0)
        assert jnp.all(result < 1.0)

    def test_deterministic(self):
        result1 = sobol_uniform_1d(50)
        result2 = sobol_uniform_1d(50)
        assert jnp.allclose(result1, result2)

    def test_start_index(self):
        result1 = sobol_uniform_1d(5, start_index=1)
        result2 = sobol_uniform_1d(5, start_index=6)
        assert not jnp.allclose(result1, result2)


class TestSobolNormal1d:
    def test_output_shape(self):
        n = 100
        result = sobol_normal_1d(n)
        assert result.shape == (n,)

    def test_deterministic(self):
        result1 = sobol_normal_1d(50)
        result2 = sobol_normal_1d(50)
        assert jnp.allclose(result1, result2)

    def test_start_index(self):
        result1 = sobol_normal_1d(5, start_index=1)
        result2 = sobol_normal_1d(5, start_index=6)
        assert not jnp.allclose(result1, result2)
