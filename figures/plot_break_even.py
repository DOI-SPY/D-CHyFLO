"""Plot the expanded deterministic break-even diagnostic."""

import matplotlib.pyplot as plt

from _plot_utils import finish, table


def main():
    data = table("break_even_summary.csv")
    plt.bar(data["diagnostic"], data["break_even_value"])
    plt.ylabel("Break-even value")
    plt.xticks(rotation=15, ha="right")
    return finish("break_even.png")


if __name__ == "__main__":
    main()
