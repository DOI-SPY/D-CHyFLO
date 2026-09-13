"""Plot the annual cumulative net-carbon ledger."""

import matplotlib.pyplot as plt

from _plot_utils import finish, table


def main():
    data = table("lifecycle_ledger.csv")
    plt.plot(data["year"], data["cumulative_net_carbon_tco2e"])
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Year")
    plt.ylabel("Cumulative net carbon (t CO2e)")
    return finish("lifecycle_trajectory.png")


if __name__ == "__main__":
    main()
