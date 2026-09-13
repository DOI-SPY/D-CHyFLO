"""Plot the example Marginal Water-Carbon Value summary."""

import matplotlib.pyplot as plt

from _plot_utils import finish, table


def main():
    data = table("mcv_summary.csv")
    plt.bar(data["metric"], data["mcv_kgco2_per_m3"])
    plt.ylabel("MCV (kg CO2/m3)")
    return finish("mcv_patterns.png")


if __name__ == "__main__":
    main()
