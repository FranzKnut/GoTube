# computes the jacobian and the metric for a given model

import jax.numpy as jnp
import numpy as np
from jax import jacfwd, jacrev, jit

from scipy.linalg import eigh
from numpy.linalg import inv
import gotube.benchmarks as bm


class FunctionDynamics:
    def __init__(self, model):
        self.model = model

        x = jnp.ones(self.model.dim)
        if jnp.sum(jnp.abs(self.model.fdyn(0.0, x) - self.model.fdyn(1.0, x))) > 1e-8:
            # https://github.com/google/jax/issues/47
            raise ValueError("Only time-invariant systems supported currently")
        self._cached_f_jac = jit(jacrev(lambda x: self.model.fdyn(0.0, x)))

    def f_jac_at(self, t, x):
        return jnp.array(self._cached_f_jac(x))

    def metric(self, Fmid, ellipsoids):
        if ellipsoids:
            A1inv = Fmid
            A1 = inv(A1inv)
            M1 = inv(A1inv @ A1inv.T)

            W, v = eigh(M1)

            W = abs(W)  # to prevent nan-errors

            semiAxes = 1 / np.sqrt(W)  # needed to compute volume of ellipse

        else:
            A1 = np.eye(Fmid.shape[0])
            M1 = np.eye(Fmid.shape[0])
            semiAxes = np.array([1])

        return M1, A1, semiAxes.prod()

if __name__ == "__main__":
    fdyn = FunctionDynamics(bm.CartpoleCTRNN())

    print(fdyn.f_jac_at(0, fdyn.model.cx))
