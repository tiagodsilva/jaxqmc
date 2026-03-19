import jax.numpy as jnp
import matplotlib  # noqa: F401
import matplotlib.pyplot as plt

from jaxqmc import sobol_normal_1d, sobol_uniform_1d

matplotlib.use("Agg")

n = 1024

u_samples = sobol_uniform_1d(n)
z_samples = sobol_normal_1d(n)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.hist(jnp.asarray(u_samples), bins=50, edgecolor="black", alpha=0.7)
ax1.set_title("Sobol Uniform 1D")
ax1.set_xlabel("Value")
ax1.set_ylabel("Frequency")
ax1.set_xlim(0, 1)

ax2.hist(jnp.asarray(z_samples), bins=50, edgecolor="black", alpha=0.7, color="orange")
ax2.set_title("Sobol Normal 1D")
ax2.set_xlabel("Value")
ax2.set_ylabel("Frequency")

plt.tight_layout()
plt.savefig("examples/uniform_normal_histograms.png", dpi=150)
print("Saved examples/uniform_normal_histograms.png")
