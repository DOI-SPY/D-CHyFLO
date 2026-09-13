"""Plot local MEF deviation across the perturbation scan."""

import matplotlib.pyplot as plt

from _plot_utils import finish, table


def main():
    data = table("mef_validity.csv")
    plt.plot(data["shift_mw"], 100 * data["relative_deviation"], marker="o")
    plt.axhline(10, color="black", linestyle="--", label="10% threshold")
    plt.xlabel("Grid injection shift (MW)")
    plt.ylabel("Local MEF deviation (%)")
    plt.legend()
    return finish("mef_validity.png")


if __name__ == "__main__":
    main()
