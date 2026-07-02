"""Matplotlib schematic of the R1C1 room-heating model (electrical analogy)."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


def _resistor(ax, x0, x1, y, label):
    """Draw a horizontal zigzag resistor with straight leads between x0 and x1."""
    lead = 0.2 * (x1 - x0)
    xs = np.linspace(x0 + lead, x1 - lead, 13)
    ys = np.full(xs.shape, float(y))
    ys[1:-1:2] += 0.16
    ys[2:-1:2] -= 0.16
    ax.plot([x0, x0 + lead], [y, y], color="k", lw=1.5)
    ax.plot(xs, ys, color="k", lw=1.5)
    ax.plot([x1 - lead, x1], [y, y], color="k", lw=1.5)
    ax.annotate(label, ((x0 + x1) / 2, y + 0.42), ha="center", fontsize=12)


def _capacitor_to_ground(ax, x, y_top, label):
    """Draw a capacitor from (x, y_top) down to a ground symbol."""
    plate_y = y_top - 0.55
    ax.plot([x, x], [y_top, plate_y], color="k", lw=1.5)
    ax.plot([x - 0.35, x + 0.35], [plate_y, plate_y], color="k", lw=2.5)
    ax.plot([x - 0.35, x + 0.35], [plate_y - 0.16, plate_y - 0.16], color="k", lw=2.5)
    ax.plot([x, x], [plate_y - 0.16, plate_y - 0.42], color="k", lw=1.5)
    for i, w in enumerate([0.28, 0.18, 0.08]):
        ax.plot(
            [x - w, x + w],
            [plate_y - 0.42 - 0.09 * i] * 2,
            color="k",
            lw=1.5,
        )
    ax.annotate(label, (x + 0.55, plate_y - 0.08), fontsize=12)


def draw_rc_thermal() -> Figure:
    """Draw the R1C1 network: T_out —[R]— T with capacitance C, heated by q.

    Thermal-electrical analogy: temperature = voltage, heat flow = current.
    The outdoor temperature is a (forecast) source node, the room is one
    temperature node with a heat capacity, and the heater injects q directly
    into the room node.
    """
    fig, ax = plt.subplots(figsize=(8.0, 3.2))

    y = 2.0

    # Outdoor temperature node (source).
    ax.add_patch(Circle((1.1, y), 0.1, fill=False, lw=1.5))
    ax.annotate(
        "$T_{\\mathrm{out}}$\n(forecast)", (1.1, y + 0.42), ha="center", fontsize=11
    )

    # Thermal resistance between outdoors and room.
    _resistor(ax, 1.2, 4.7, y, "$R$")

    # Room temperature node (the state).
    ax.add_patch(Circle((4.8, y), 0.07, color="k"))
    ax.annotate("$T$ (state)", (4.8, y + 0.42), ha="center", fontsize=11)

    # Room heat capacity.
    _capacitor_to_ground(ax, 4.8, y - 0.07, "$C$")

    # Heater input.
    ax.add_patch(
        FancyArrowPatch(
            (7.3, y), (5.0, y), arrowstyle="-|>", mutation_scale=18,
            color="tab:red", lw=2,
        )
    )
    ax.annotate(
        "$q$ [kW] (control)", (7.45, y), va="center", color="tab:red", fontsize=11
    )

    # Dashed box marking the room.
    ax.add_patch(
        FancyBboxPatch(
            (4.0, y - 1.75), 1.9, 2.35,
            boxstyle="round,pad=0.08", fill=False, ls="--", ec="gray",
        )
    )
    ax.annotate("room", (4.15, y - 1.65), color="gray", fontsize=10)

    ax.set_xlim(0.0, 9.3)
    ax.set_ylim(0.0, 3.1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig
