"""Plot marginal carbon benefit; the response transition is state dependent."""

import matplotlib.pyplot as plt

from _plot_utils import finish, table


def main():
    data = table("capacity_summary.csv")
    for ratio, group in data.groupby("dc_ac_ratio"):
        feasible = group[group["geometry_screen_pass"]]
        plt.plot(
            feasible["capacity_mwac"],
            feasible["marginal_avoided_kgco2_per_added_mw"],
            marker="o",
            label=f"{ratio:g}",
        )
    plt.xlabel("FPV capacity (MWac)")
    plt.ylabel("Marginal avoided carbon per added MW")
    plt.legend(title="DC/AC")
    return finish("capacity_knee.png")


if __name__ == "__main__":
    main()
