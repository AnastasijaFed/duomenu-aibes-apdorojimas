from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator

input_file = "duomenys/tiriamoji.csv"
output_folder = Path("duomenys/klasiu_palyginimas")

class_column = "klase"
classes = ["N", "L", "R", "V", "A"]

numeric_features = [
    "RR_pre", "RR_post", "RR_vid", "RR_sant_vid", "RR_sant_post",
    "PR", "QRS", "QT", "QTc",
    "P_amp", "T_amp", "ST", "QRS_max", "QRS_min",
]

palette = ["#d62728", "#9467bd", "#98df8a", "#f7b6d2", "#aec7e8"]   # red, purple, light green, light pink, light blue
title_size = 16
label_size = 14
tick_size = 13
legend_size = 13
value_size = 10


def medians_by_class(data):
    return data.groupby(class_column)[numeric_features].median().T[classes]


def decimal_comma(value, decimals):
    return f"{value:.{decimals}f}".replace(".", ",")


def draw_bars(axis, medians, features, colours, show_values=True, group_width=0.8):
    bar_width = group_width / len(features)
    positions = np.arange(len(classes))

    for number, (feature, colour) in enumerate(zip(features, colours)):
        bar_positions = positions - group_width / 2 + bar_width * (number + 0.5)
        values = medians.loc[feature, classes]

        bars = axis.bar(bar_positions, values, width=bar_width, color=colour,
                        edgecolor="#333333", linewidth=0.8, label=feature)

        if show_values:
            decimals = 0 if abs(values).max() >= 100 else 2
            labels = [decimal_comma(value, decimals) for value in values]
            axis.bar_label(bars, labels=labels, padding=3, fontsize=value_size)

    axis.set_facecolor("white")
    axis.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    axis.set_axisbelow(True)
    for side in axis.spines.values():
        side.set_visible(False)
    axis.tick_params(length=0)

    axis.set_xticks(positions, classes, fontsize=tick_size + 2)
    axis.tick_params(axis="y", labelsize=tick_size)
    axis.set_xlabel("Dūžio klasė", fontsize=label_size)


def use_decimal_comma_on_y(axis, decimals):
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: decimal_comma(value, decimals)))


def save(figure, file_name):
    plt.tight_layout()
    plt.savefig(output_folder / file_name, dpi=200)
    plt.close(figure)


def plot_rr_pre(medians):
    figure, axis = plt.subplots(figsize=(11, 6))

    draw_bars(axis, medians, ["RR_pre"], palette[:1], group_width=0.45)
    axis.set_ylabel("Mediana, ms", fontsize=label_size)
    axis.set_title("RR intervalas prieš dūžį", fontsize=title_size)
    axis.margins(y=0.12)

    save(figure, "rr_pre.png")


def plot_rr_ratios(medians):
    figure, axis = plt.subplots(figsize=(13, 6))

    draw_bars(axis, medians, ["RR_sant_vid", "RR_sant_post"], palette[1:3])
    axis.axhline(1, color="#616161", linestyle="--", linewidth=1.2)
    use_decimal_comma_on_y(axis, 1)
    axis.set_ylabel("Mediana", fontsize=label_size)
    axis.set_title("RR intervalų santykiai (punktyrinė linija – santykis 1)", fontsize=title_size)
    axis.margins(y=0.12)
    axis.legend(title="Požymis", bbox_to_anchor=(1.01, 0.5), loc="center left",
                frameon=False, fontsize=legend_size, title_fontsize=legend_size + 1)

    save(figure, "rr_santykiai.png")


def plot_durations(medians):
    figure, axis = plt.subplots(figsize=(13, 6))

    draw_bars(axis, medians, ["QRS", "QTc", "PR"], palette[:3])
    axis.set_ylabel("Mediana, ms", fontsize=label_size)
    axis.set_title("Bangų trukmės", fontsize=title_size)
    axis.margins(y=0.12)
    axis.legend(title="Požymis", bbox_to_anchor=(1.01, 0.5), loc="center left",
                frameon=False, fontsize=legend_size, title_fontsize=legend_size + 1)

    save(figure, "trukmes.png")


def plot_amplitudes(medians):
    figure, axis = plt.subplots(figsize=(13, 6))

    features = ["P_amp", "T_amp", "ST", "QRS_max", "QRS_min"]
    draw_bars(axis, medians, features, palette[:5], show_values=False)

    axis.axhline(0, color="#616161", linestyle="--", linewidth=1.2)
    axis.yaxis.set_major_locator(MultipleLocator(0.25))
    use_decimal_comma_on_y(axis, 2)
    axis.set_ylabel("Mediana, signalo vnt.", fontsize=label_size)
    axis.set_title("Amplitudės", fontsize=title_size)
    axis.legend(title="Požymis", bbox_to_anchor=(1.01, 0.5), loc="center left",
                frameon=False, fontsize=legend_size, title_fontsize=legend_size + 1)

    save(figure, "amplitudes.png")


def main():
    output_folder.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(input_file)
    medians = medians_by_class(data)

    plot_rr_pre(medians)
    plot_rr_ratios(medians)
    plot_durations(medians)
    plot_amplitudes(medians)

    print("Požymių medianos pagal klasę:")
    print(medians.round(3))


if __name__ == "__main__":
    main()