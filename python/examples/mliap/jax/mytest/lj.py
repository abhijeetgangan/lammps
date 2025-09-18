import jax
import jax.numpy as jnp
from functools import partial
from lammps.mliap.mliap_unified_abc import MLIAPUnified
import numpy as np
import cupy

class MLIAPLJ(MLIAPUnified):
    """Test implementation of Lennard-Jones MLIAPUnified."""

    def __init__(self, element_types, epsilon=1.0, sigma=1.0):
        super().__init__()
        self.element_types = element_types
        self.epsilon = epsilon
        self.sigma = sigma
        self.rcutfac = 2.5 * sigma / 2.0
        self.dtype = jnp.float32
        self.ndescriptors = 1
        self.nparams = 2
        self.num_species = len(element_types)
        self.device = "gpu"
        
    def compute_gradients(self, data):
        """Test compute_gradients."""

    def compute_descriptors(self, data):
        """Test compute_descriptors."""

    def compute_forces(self, data):
        """Test compute_forces."""
        total_energy, fij = self._compute_pair_ef(data)
        data.energy = total_energy
        data.update_pair_forces_gpu(cupy.asarray(fij.astype(jnp.float64)))

    def _compute_pair_ef(self, data):
        """Compute pair energy and forces."""
        rij = jnp.asarray(data.rij).astype(self.dtype)        
        dr = jnp.linalg.vector_norm(rij, axis=1, keepdims=True)

        @jax.jit
        def compute_energy_and_force(dr):
            r2inv = 1.0 / dr**2
            r6inv = r2inv * r2inv * r2inv
            lj1 = 4.0 * self.epsilon * self.sigma**12
            lj2 = 4.0 * self.epsilon * self.sigma**6
            eij = r6inv * (lj1 * r6inv - lj2)
            fij = r6inv * (3.0 * lj2 - 6.0 * lj2 * r6inv) * r2inv
            fij = fij[:, np.newaxis] * rij
            return eij, fij

        eij, fij = jax.vmap(compute_energy_and_force)(dr)
        total_energy = jnp.sum(eij)

        return total_energy, fij