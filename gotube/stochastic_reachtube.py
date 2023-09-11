# optimization problem
from functools import partial
import numpy as np
import jax.numpy as jnp
from jax.experimental.ode import odeint
from jax import device_put, devices, jacrev, pmap, vmap, jit

from scipy.special import gamma
from scipy.linalg import eigh
from numpy.linalg import inv

# own files
import gotube.benchmarks as bm
import gotube.polar_coordinates as pol
import gotube.dynamics as dynamics


def create_aug_state_cartesian(x, F):
    return jnp.concatenate((jnp.array([x]), F)).reshape(-1)  # reshape to row vector


class StochasticReachtube:
    def __init__(
        self,
        system: bm.BaseSystem = bm.CartpoleCTRNN(None),
        time_horizon: float = 10.0,  # time_horizon until which the reachtube should be constructed
        profile: bool = False,
        time_step: float = 0.1,  # ReachTube construction
        h_metric: float = 0.05,  # time_step for metric computation
        max_step_metric: float = 0.00125,  # maximum time_step for metric computation
        max_step_optim: float = 0.1,  # maximum time_step for optimization
        batch: int = 1,  # number of initial points for vectorization
        num_gpus: int = 1,  # number of GPUs for parallel computation
        fixed_seed=False,  # specify whether a fixed seed should be used (only for comparing different algorithms)
        atol: float = 1e-10,  # absolute tolerance of integration
        rtol: float = 1e-10,  # relative tolerance of integration
        plot_grid: int = 50,
        mu: float = 1.5,
        gamma: float = 0.01,
        radius: bool = False,
    ):
        """Construct a stochastic reachtube for a given model

        Parameters
        ----------
        model : bm.BaseModel, optional
            _description_, by default bm.CartpoleCTRNN(None)
        time_horizon : float, optional
            _description_, by default 10.0
        time_step : float, optional
            _description_, by default 0.1
        atol : float, optional
            _description_, by default 1e-10
        mu : float, optional
            _description_, by default 1.5
        gamma : float, optional
            _description_, by default 0.01
        radius : bool, optional
            _description_, by default False
        """

        self.rad_t0 = None
        self.cx_t0 = None
        self.t0 = None
        self.cur_rad = None
        self.cur_cx = None
        self.cur_time = None
        self.A0inv = None
        self.A1inv = None
        self.A1 = None
        self.M1 = None
        self.time_step = jnp.minimum(time_step, time_horizon)
        self.profile = profile
        self.h_metric = jnp.minimum(h_metric, time_step)
        self.max_step_metric = jnp.minimum(max_step_metric, self.h_metric)
        self.max_step_optim = jnp.minimum(max_step_optim, self.time_step)
        self.time_horizon = time_horizon
        self.batch = batch
        self.num_gpus = num_gpus
        self.fixed_seed = fixed_seed
        self.atol = atol
        self.rtol = rtol
        self.plotGrid = plot_grid
        self.mu = mu
        self.gamma = gamma

        self.model = system
        self.init_model()

        x = jnp.ones(self.model.dim)
        if jnp.sum(jnp.abs(self.model.fdyn(0.0, x) - self.model.fdyn(1.0, x))) > 1e-8:
            # https://github.com/google/jax/issues/47
            raise ValueError("Only time-invariant systems supported currently")
        self._cached_f_jac = jit(jacrev(lambda x: self.model.fdyn(0.0, x)))

        self.init_metric()

        self.f_jac_at = dynamics.FunctionDynamics(system).f_jac_at

    def init_metric(self):
        self.M1 = np.eye(self.model.dim)
        self.A1 = np.eye(self.model.dim)
        self.A1inv = np.eye(self.model.dim)
        self.A0inv = np.eye(self.model.dim)

    def init_model(self):
        self.cur_time = 0
        self.cur_cx = self.model.cx
        self.cur_rad = self.model.rad
        self.t0 = 0
        self.cx_t0 = self.model.cx
        self.rad_t0 = self.model.rad

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

    def compute_volume(self, semiAxes_product=None):
        if semiAxes_product is None:
            semiAxes_product = 1
        volC = gamma(self.model.dim / 2.0 + 1) ** -1 * jnp.pi ** (
                self.model.dim / 2.0
        )  # volume constant for ellipse and ball
        return volC * self.cur_rad ** self.model.dim * semiAxes_product

    def propagate_center_point(self, time_range):
        cx_jax = self.model.cx.reshape(1, self.model.dim)
        F = jnp.eye(self.model.dim)
        # put aug_state in CPU, as it is faster for odeint than GPU
        aug_state = device_put(jnp.concatenate((cx_jax, F)).reshape(1, -1), device=devices("cpu")[0])
        sol = odeint(
            self.aug_fdyn_jax_no_pmap,
            aug_state,
            time_range,
            atol=self.atol,
            rtol=self.rtol,
        )
        cx, F = vmap(self.reshape_aug_state_to_matrix)(sol)
        return cx, F

    def compute_metric_and_center(self, time_range, ellipsoids):
        print(f"Propagating center point for {time_range.shape[0] - 1} timesteps")
        cx_timeRange, F_timeRange = self.propagate_center_point(time_range)
        A1_timeRange = np.eye(self.model.dim).reshape(1, self.model.dim, self.model.dim)
        M1_timeRange = np.eye(self.model.dim).reshape(1, self.model.dim, self.model.dim)
        semiAxes_prod_timeRange = np.array([1])

        print("Starting loop for creating metric")
        for idx, t in enumerate(time_range[1:]):
            M1_t, A1_t, semiAxes_prod_t = self.metric(
                F_timeRange[idx + 1, :, :], ellipsoids
            )
            A1_timeRange = np.concatenate(
                (A1_timeRange, A1_t.reshape(1, self.model.dim, self.model.dim)), axis=0
            )
            M1_timeRange = np.concatenate(
                (M1_timeRange, M1_t.reshape(1, self.model.dim, self.model.dim)), axis=0
            )
            semiAxes_prod_timeRange = np.append(
                semiAxes_prod_timeRange, semiAxes_prod_t
            )

        return cx_timeRange, A1_timeRange, M1_timeRange, semiAxes_prod_timeRange

    def reshape_aug_state_to_matrix(self, aug_state):
        aug_state = aug_state.reshape(-1, self.model.dim)  # reshape to matrix
        x = aug_state[:1][0]
        F = aug_state[1:]
        return x, F

    def reshape_aug_fdyn_return_to_vector(self, fdyn_return, F_return):
        return jnp.concatenate((jnp.array([fdyn_return]), F_return)).reshape(-1)

    @partial(jit, static_argnums=(0,))
    def aug_fdyn(self, t=0, aug_state=0):
        x, F = self.reshape_aug_state_to_matrix(aug_state)
        fdyn_return = self.model.fdyn(t, x)
        F_return = jnp.matmul(self.f_jac_at(t, x), F)
        return self.reshape_aug_fdyn_return_to_vector(fdyn_return, F_return)

    def aug_fdyn_jax_no_pmap(self, aug_state=0, t=0):
        return vmap(self.aug_fdyn, in_axes=(None, 0))(t, aug_state)

    def fdyn_jax_no_pmap(self, x=0, t=0):
        return vmap(self.model.fdyn, in_axes=(None, 0))(t, x)

    def aug_fdyn_jax(self, aug_state=0, t=0):
        return pmap(vmap(self.aug_fdyn, in_axes=(None, 0)), in_axes=(None, 0))(t, aug_state)

    def fdyn_jax(self, x=0, t=0):
        return pmap(vmap(self.model.fdyn, in_axes=(None, 0)), in_axes=(None, 0))(t, x)

    def create_aug_state(self, polar, rad_t0, cx_t0):
        x = jnp.array(
            pol.polar2cart_euclidean_metric(rad_t0, polar, self.A0inv) + cx_t0
        )
        F = jnp.eye(self.model.dim)

        aug_state = jnp.concatenate((jnp.array([x]), F)).reshape(
            -1
        )  # reshape to row vector

        return aug_state, x

    def one_step_aug_integrator(self, x, F):
        aug_state = pmap(vmap(create_aug_state_cartesian))(x, F)
        sol = odeint(
            self.aug_fdyn_jax,
            aug_state,
            jnp.array([0, self.time_step]),
            atol=self.atol,
            rtol=self.rtol,
        )
        x, F = pmap(vmap(self.reshape_aug_state_to_matrix))(sol[-1])
        return x, F

    def aug_integrator(self, polar, step=None):
        if step is None:
            step = self.cur_time

        rad_t0 = self.rad_t0
        cx_t0 = self.cx_t0

        aug_state, initial_x = pmap(vmap(self.create_aug_state, in_axes=(0, None, None)), in_axes=(0, None, None))(
            polar, rad_t0, cx_t0
        )
        sol = odeint(
            self.aug_fdyn_jax,
            aug_state,
            jnp.array([0, step]),
            atol=self.atol,
            rtol=self.rtol,
        )
        x, F = pmap(vmap(self.reshape_aug_state_to_matrix))(sol[-1])
        return x, F, initial_x

    def aug_integrator_neg_dist(self, polar):
        x, F, initial_x = self.aug_integrator(polar)
        neg_dist = pmap(vmap(self.neg_dist_x))(x)
        return x, F, neg_dist, initial_x

    def one_step_aug_integrator_dist(self, x, F):
        x, F = self.one_step_aug_integrator(x, F)
        neg_dist = pmap(vmap(self.neg_dist_x))(x)
        return x, F, -neg_dist

    def neg_dist_x(self, xt):
        dist = jnp.linalg.norm(jnp.matmul(self.A1, xt - self.cur_cx))
        return -dist
