"""Plotting script."""

import matplotlib.pyplot as plt
import pandas as pd

FILE_PATH = "output/i4b_baseline/val_timeseries_step0.csv"

if __name__ == "__main__":
    df = pd.read_csv(FILE_PATH)

    # Normalize all time columns by solver_time_tot
    time_cols = [c for c in df.columns if "time" in c]
    for col in time_cols:
        df[col + "_normalized"] = df[col] / df["solver_time_tot"]

    # Figure 1: room temperature vs setpoints
    n_subplots = 3
    plt.subplot(n_subplots, 1, 1)
    plt.plot(df["step"], df["T_room"], label="T_room")
    plt.plot(df["step"], df["T_set_upper"], label="T_set_upper", linestyle="--")
    plt.plot(df["step"], df["T_set_lower"], label="T_set_lower", linestyle="--")
    plt.xlabel("Step")
    plt.ylabel("T (°C)")
    plt.title("Room Temperature vs Setpoint")
    plt.legend()
    plt.grid(True)
    plt.subplot(n_subplots, 1, 2)
    plt.plot(df["step"], df["T_amb"], label="T_amb")
    plt.xlabel("Step")
    plt.ylabel("T (°C)")
    plt.title("Ambient Temperature")
    plt.legend()
    plt.grid(True)
    plt.subplot(n_subplots, 1, 3)
    plt.plot(df["step"], df["T_hp_sup"], label="T_hp_sup")
    plt.xlabel("Step")
    plt.ylabel("T_hp_sup (°C)")
    plt.title("Control Input")
    plt.legend()
    plt.grid(True)

    # Figure 2: normalised solver time breakdown
    norm_cols = [c for c in df.columns if "time" in c and c.endswith("_normalized")]
    fig, ax = plt.subplots(2, 1, sharex=True)
    for col in time_cols:
        ax[0].plot(df["step"], df[col], label=col)
    ax[0].set_ylabel("Solver time (s)")
    ax[0].legend(fontsize=7)
    ax[0].grid(True)
    for col in norm_cols:
        ax[1].plot(df["step"], df[col], label=col)
    ax[1].set_xlabel("Step")
    ax[1].set_ylabel("Fraction of solver_time_tot")
    ax[1].set_title("Normalised solver time breakdown")
    ax[1].legend(fontsize=7)
    ax[1].grid(True)

    plt.show()
