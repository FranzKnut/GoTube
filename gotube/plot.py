# plotting the outputs from GoTube (ellipse, circle, intersections etc.)
import jax
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


def draw_ellipse(ellipse, color, alph, axis_3d):
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

    axis_3d.plot(xs=X[0], ys=X[1], zs=ellipse[0], color=color, alpha=alph)


def plot_ellipses(
        time_horizon, dim, projection_axes, data_ellipse, axis_3d, skip_reachsets=1, color='magenta', alph=0.8
):
    # permutation matrix to project on axis1 and axis2
    P = np.eye(dim)
    P[:, [0, projection_axes[0]]] = P[:, [projection_axes[0], 0]]
    P[:, [1, projection_axes[1]]] = P[:, [projection_axes[1]]]

    count = skip_reachsets

    for ellipse in data_ellipse[1:]:

        if count != skip_reachsets:
            count += 1
            continue

        count = 1

        ellipse2 = ellipse

        # create ellipse plotting values for 2d projection
        # https://math.stackexchange.com/questions/2438495/showing-positive-definiteness-in-the-projection-of-ellipsoid
        # construct ellipse2 to have a 2-dimensional ellipse as an input to
        # ellipse_plot

        if dim > 2:
            center = ellipse[1: dim + 1]
            ellipse2[1] = center[projection_axes[0]]
            ellipse2[2] = center[projection_axes[1]]
            radius_ellipse = ellipse[dim + 1]
            m1 = np.reshape(ellipse[dim + 2:], (dim, dim))
            m1 = m1 / radius_ellipse ** 2
            m1 = P.transpose() @ m1 @ P  # permutation to project on chosen axes
            ellipse2[3] = 1  # because radius is already in m1

            # plot ellipse onto axis1-axis2 plane
            J = m1[0:2, 0:2]
            K = m1[2:, 2:]
            L = m1[2:, 0:2]
            m2 = J - L.transpose() @ inv(K) @ L
            ellipse2[4:8] = m2.reshape(1, -1)

        draw_ellipse(ellipse2[0:8], color, alph, axis_3d)

        if ellipse[0] >= time_horizon:
            break


def get_random_trace_samples(
        reachtube: reach.StochasticReachtube,
        timestep_granularity=0.01
):
    rd_polar = pol.init_random_phi(reachtube.model.dim, reachtube.sample_count)
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
        axis_3d: plt.axes,
        projection_axes: tuple = (0, 1)
):
    for sample_index in range(reachtube.sample_count):
        axis_3d.plot(
            xs=states_at_timesteps[:, sample_index, projection_axes[0]],
            ys=states_at_timesteps[:, sample_index, projection_axes[1]],
            zs=timesteps,
            color="k",
            linewidth=1,
        )


def reshape_samples_for_pickle(timesteps, states_at_timesteps):
    return {
        "xs": np.array(states_at_timesteps[:, len(timesteps), projection_axes[0]]),
        "ys": np.array(states_at_timesteps[:, len(timesteps), projection_axes[1]]),
        "zs": np.array(timesteps),
    }


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


def load_gotube_output_filepath():
    parser = configparser.ConfigParser()
    parser.read("gotube/config.ini")
    file_config = parser["files"]
    return file_config["output_directory"] + str(args.output_number) + file_config["output_file"]


def save_pickle_dict_file():
    os.makedirs("plot_obj", exist_ok=True)
    for i in range(1000):
        filename = f"plot_obj/plot_{i:03d}.pkl"
        if not os.path.isfile(filename):
            with open(filename, "wb") as f:
                pickle.dump(pickle_dict, f, protocol=pickle.DEFAULT_PROTOCOL)
            break


def prepare_figure_and_axes():
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


if __name__ == "__main__":
    args = parse_arguments()
    gotube_output_filepath = load_gotube_output_filepath()
    ellipse_data = np.loadtxt(gotube_output_filepath)
    projection_axes = (args.axis1, args.axis2)

    reachtube = reach.StochasticReachtube(
        system=bm.get_model(args.benchmark, args.radius),
        time_horizon=args.time_horizon,
        time_step=args.time_step,
        samples=args.samples,
    )

    fig, ax = prepare_figure_and_axes()

    timesteps, states_at_timesteps = get_random_trace_samples(reachtube)
    plot_traces(timesteps, states_at_timesteps, ax, projection_axes)

    pickle_dict = dict(
        xs=np.array(states_at_timesteps[:, len(timesteps), projection_axes[0]]),
        ys=np.array(states_at_timesteps[:, len(timesteps), projection_axes[1]]),
        zs=timesteps,
        time_horizon=args.time_horizon,
        dim=reachtube.model.dim,
        projection_axes=projection_axes,
        ellipse_data=ellipse_data
    )
    save_pickle_dict_file()

    plot_ellipses(
        args.time_horizon,
        reachtube.model.dim,
        projection_axes,
        ellipse_data,
        axis_3d=ax,
    )
    plt.show()
