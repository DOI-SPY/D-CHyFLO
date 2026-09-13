"""Plot the CF0-CF3 operational carbon contribution shares."""

import matplotlib.pyplot as plt

from _plot_utils import finish, table


def main():
    data = table("operational_attribution.csv")
    plt.bar(data["contribution"], data["share_percent"])
    plt.ylabel("Share of operational carbon benefit (%)")
    plt.xticks(rotation=20, ha="right")
    return finish("operational_attribution.png")


if __name__ == "__main__":
    main()
