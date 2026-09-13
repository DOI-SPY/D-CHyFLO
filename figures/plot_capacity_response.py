"""Plot operational avoided carbon across FPV capacities."""

import matplotlib.pyplot as plt

from _plot_utils import finish, table


def main():
    data = table("capacity_summary.csv")
    for ratio, group in data.groupby("dc_ac_ratio"):
        feasible = group[group["geometry_screen_pass"]]
        plt.plot(
            feasible["capacity_mwac"], feasible["operational_avoided_kgco2"], label=f"{ratio:g}"
        )
    plt.xlabel("FPV capacity (MWac)")
    plt.ylabel("Example operational avoided carbon (kg CO2)")
    plt.legend(title="DC/AC")
    return finish("capacity_response.png")


if __name__ == "__main__":
    main()
