# plotting the outputs from GoTube (ellipse, circle, intersections etc.)
import jax
import matplotlib.axes
import numpy
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from numpy.linalg import svd
from numpy.linalg import inv
from jax.experimental.ode import odeint

import gotube.stochastic_reachtube as reach
import gotube.benchmarks as bm
import gotube.polar_coordinates as pol
import configparser
import argparse
import pickle
import os

from jax import config, vmap

config.update("jax_enable_x64", True)


def draw_ellipse(
        ellipse: numpy.ndarray,
        color: str,
        alpha: float,
        axis_3d: matplotlib.axes.Axes
):
    c = ellipse[1:3]
    r = ellipse[3]
    M = np.reshape(ellipse[4:8], (2, 2)) / r ** 2

    # "singular value decomposition" to extract the orientation and the
    # axes of the ellipsoid
    _, D, V = svd(M)

    plot_grid = 50

    # get the major and minor axes
    a = 1 / np.sqrt(D[0])
    b = 1 / np.sqrt(D[1])

    theta = np.arange(0, 2 * np.pi + 1 / plot_grid, 1 / plot_grid)

    # parametric equation of the ellipse
    state = np.zeros((2, np.size(theta)))
    state[0, :] = a * np.cos(theta)
    state[1, :] = b * np.sin(theta)

    # coordinate transform
    X = V @ state

    X[0] += c[0]
    X[1] += c[1]

    axis_3d.plot(xs=X[0], ys=X[1], zs=ellipse[0], color=color, alpha=alpha)


def plot_ellipses(
        time_horizon: float,
        dim: int,
        projection_axes: tuple,
        data_ellipse: numpy.ndarray,
        axis_3d: matplotlib.axes.Axes,
        skip_reachsets: int = 1,
        color: str = 'magenta',
        alph: float = 0.8
):
    # permutation matrix to project on axis1 and axis2
    P = np.eye(dim)
    P[:, [0, projection_axes[0]]] = P[:, [projection_axes[0], 0]]
    P[:, [1, projection_axes[1]]] = P[:, [projection_axes[1]]]

    for original_ellipse in data_ellipse[1::skip_reachsets]:

        ellipse2d = original_ellipse

        # create ellipse plotting values for 2d projection
        # https://math.stackexchange.com/questions/2438495/showing-positive-definiteness-in-the-projection-of-ellipsoid
        # construct ellipse2d to have a 2-dimensional ellipse as an input to
        # ellipse_plot

        if dim > 2:
            center = original_ellipse[1: dim + 1]
            ellipse2d[1] = center[projection_axes[0]]
            ellipse2d[2] = center[projection_axes[1]]
            radius_ellipse = original_ellipse[dim + 1]
            m1 = np.reshape(original_ellipse[dim + 2:], (dim, dim))
            m1 = m1 / radius_ellipse ** 2
            m1 = P.transpose() @ m1 @ P  # permutation to project on chosen axes
            ellipse2d[3] = 1  # because radius is already in m1

            # plot ellipse onto axis1-axis2 plane
            J = m1[0:2, 0:2]
            K = m1[2:, 2:]
            L = m1[2:, 0:2]
            m2 = J - L.transpose() @ inv(K) @ L
            ellipse2d[4:8] = m2.reshape(1, -1)

        draw_ellipse(ellipse2d[0:8], color, alph, axis_3d)

        if original_ellipse[0] >= time_horizon:
            break


def get_random_trace_samples(
        reachtube: reach.StochasticReachtube,
        sample_count: int,
        timestep_granularity: float = 0.01,
):
    rd_polar = pol.init_random_phi(reachtube.model.dim, sample_count)
    rd_polar = jnp.reshape(rd_polar, (-1, rd_polar.shape[2]))
    rd_x = (
            vmap(pol.polar2cart, in_axes=(None, 0))(reachtube.model.rad, rd_polar)
            + reachtube.model.cx
    )
    timesteps = jnp.arange(0, reachtube.time_horizon + 1e-9, timestep_granularity)

    states_at_timesteps = odeint(
        reachtube.fdyn_jax_no_pmap,
        rd_x,
        timesteps,
        atol=reachtube.atol,
        rtol=reachtube.rtol,
    )

    return timesteps, states_at_timesteps


def plot_traces(
        timesteps: jax.Array,
        states_at_timesteps: jax.Array,
        axis_3d: matplotlib.axes,
        projection_axes: tuple = (0, 1)
):
    for sample_index in range(len(timesteps)):
        axis_3d.plot(
            xs=states_at_timesteps[:, sample_index, projection_axes[0]],
            ys=states_at_timesteps[:, sample_index, projection_axes[1]],
            zs=timesteps,
            color="k",
            linewidth=1,
        )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_step", default=0.01, type=float)
    parser.add_argument("--time_horizon", default=0.01, type=float)
    parser.add_argument("--benchmark", default="bruss")
    parser.add_argument("--output_number", default="0000")
    parser.add_argument("--samples", default=100, type=int)
    parser.add_argument("--axis1", default=0, type=int)
    parser.add_argument("--axis2", default=1, type=int)
    parser.add_argument("--radius", default=None, type=float)
    return parser.parse_args()


def load_gotube_output_filepath(file_number: str):
    parser = configparser.ConfigParser()
    parser.read("gotube/config.ini")
    file_config = parser["files"]
    return file_config["output_directory"] + str(file_number) + file_config["output_file"]


def save_pickle_dict_file(pickle_dict: dict):
    os.makedirs("plot_obj", exist_ok=True)
    for i in range(1000):
        filename = f"plot_obj/plot_{i:03d}.pkl"
        if not os.path.isfile(filename):
            with open(filename, "wb") as f:
                pickle.dump(pickle_dict, f, protocol=pickle.DEFAULT_PROTOCOL)
            break


def prepare_figure_and_axes(projection_axes: tuple):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=-10.)
    ax.tick_params(axis='both', labelsize=8)
    xlabel = 'x' + str(projection_axes[0] + 1)
    ylabel = 'x' + str(projection_axes[1] + 1)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_zlabel('Time (s)', fontsize=8)
    return fig, ax


def plot_gotube(
        system: bm.BaseSystem,
        output_number: str = "0000",
        projection_axes: tuple = (0, 1),
        time_horizon: float = 0.01,
        time_step: float = 0.01,
        samples: int = 100
):
    """Plot a GoTube output file (located in the saved_outputs directory)

        Parameters
        ----------
        system : bm.BaseSystem
            The system that was used to produce the output. (e.g. an instance of bm.CartpoleCTRNN)
        output_number : str
            The number of the GoTube run that should be plotted.
            (e.g. "0001" opens saved_outputs/0001_GoTube.txt)
        projection_axes : tuple
            Which dimensions of the dynamical system should be plotted against time.
            (e.g. (4,5) to plot the 4th dimension on the x-axis and the 5th dimension on the y-axis)
        time_horizon : float
            The time_horizon parameter that was used to produce the output.
            Determines the length in seconds of the reachtube to be constructed.
        time_step : float
            The time_step parameter that was used to produce the output.
            Intermediate time-points for which the reachtube should be constructed.
        samples : int
            The number of sample traces that should be plotted.
    """
    gotube_output_filepath = load_gotube_output_filepath(output_number)
    ellipse_data = np.loadtxt(gotube_output_filepath)

    reachtube = reach.StochasticReachtube(
        system=system,
        time_horizon=time_horizon,
        time_step=time_step
    )
    timesteps, states_at_timesteps = get_random_trace_samples(reachtube, samples)

    fig, ax = prepare_figure_and_axes(projection_axes)
    plot_traces(timesteps, states_at_timesteps, ax, projection_axes)
    plot_ellipses(
        time_horizon,
        reachtube.model.dim,
        projection_axes,
        ellipse_data,
        axis_3d=ax,
    )
    plt.show()
    save_pickle_dict_file(dict(
        xs=np.array(states_at_timesteps[:, len(timesteps), projection_axes[0]]),
        ys=np.array(states_at_timesteps[:, len(timesteps), projection_axes[1]]),
        zs=timesteps,
        time_horizon=time_horizon,
        dim=reachtube.model.dim,
        projection_axes=projection_axes,
        ellipse_data=ellipse_data
    ))


if __name__ == "__main__":
    args = parse_arguments()
    projection_axes = (args.axis1, args.axis2)
    system = bm.get_model(args.benchmark, args.radius)
    plot_gotube(
        system,
        args.output_number,
        projection_axes,
        args.time_horizon,
        args.time_step,
        args.samples
    )
