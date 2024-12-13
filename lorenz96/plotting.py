import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


def plot_trajectory_3d(snapshots, plotting_indexes):
    """Plots 3d phase space trajectories."""

    if len(plotting_indexes) != 3:
        raise ValueError("plotting_indexes must have length 3.")

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot(
        snapshots[plotting_indexes[0], :],
        snapshots[plotting_indexes[1], :],
        snapshots[plotting_indexes[2], :],
        )
    ax.set_xlabel(fr"$x_{plotting_indexes[0]}$")
    ax.set_ylabel(fr"$x_{plotting_indexes[1]}$")
    ax.set_zlabel(fr"$x_{plotting_indexes[2]}$")
    plt.show()


def plot_trajectory_2d(snapshots, plotting_indexes):
    """Plots 2d phase space trajectories."""

    if len(plotting_indexes) != 2:
        raise ValueError("plotting_indexes must have length 2.")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(
        snapshots[plotting_indexes[0], :],
        snapshots[plotting_indexes[1], :],
        )
    ax.set_xlabel(fr"$x_{plotting_indexes[0]}$")
    ax.set_ylabel(fr"$x_{plotting_indexes[1]}$")
    plt.show()


def plot_trajectory_1d(snapshots, times, plotting_indexes):
    """Plots 1d time series."""

    fig, ax = plt.subplots(
        len(plotting_indexes), 1,
        figsize=(6, 1+len(plotting_indexes)),
        sharex=True)
    for ax_index, plotting_index in enumerate(plotting_indexes):
        ax[ax_index].plot(
            times,
            snapshots[plotting_index, :],
            )
        ax[ax_index].set_ylabel(fr"$x_{plotting_index}$")
    ax[-1].set_xlabel("time")
    plt.show()
