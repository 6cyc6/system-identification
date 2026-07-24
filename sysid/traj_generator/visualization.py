"""Trajectory-generator plotting helpers."""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def draw_spectrum(x, fs):
    """Plot the power spectral density for every joint signal."""
    x = np.asarray(x)
    njoints = x.shape[1]
    _, axes = plt.subplots(1)
    for joint_index in range(njoints):
        frequency, power = signal.periodogram(x[:, joint_index], fs)
        axes.semilogy(frequency, power)
        axes.set_xlabel("frequency [Hz]")
        axes.set_ylabel("PSD [V**2/Hz]")
        axes.set_title(f"Joint{joint_index + 1}")
    axes.legend([f"Joint{joint_index + 1}" for joint_index in range(njoints)])


def vis_compare_seqs(t, seqs, legends, labels, mode=None):
    """Plot one or more sequences for each robot joint."""
    njoints = seqs[0].shape[1]
    fig, axes = plt.subplots(njoints, sharex=True)

    for joint_index in range(njoints):
        axes_i = axes[joint_index] if njoints != 1 else axes
        for sequence_index, sequence in enumerate(seqs):
            values = sequence[:, joint_index]
            if mode is None:
                axes_i.plot(t[sequence_index], values)
            else:
                axes_i.plot(t[sequence_index], values, ".")
        axes_i.set_ylabel(f"Joint {joint_index + 1}")

    plt.xlabel(labels[0])
    if njoints != 1:
        fig.align_ylabels(axes[:])
        legend = axes[-1].legend(legends)
    else:
        fig.align_ylabels(axes)
        legend = axes.legend(legends)
    legend.get_frame().set_edgecolor("b")
    legend.get_frame().set_linewidth(0.0)
