# Outlier visualisation: three figures that each answer one question.
#   1. Are the RR interval outliers real premature beats?
#   2. In which features and classes are the outliers concentrated?
#   3. Is one ECG record responsible for the most extreme values?

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # figures are written to files, no window is opened

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

cleaned_file = "duomenys/tiriamoji_sutvarkyta.csv"
output_folder = Path("duomenys/isskirtys")

class_column = "klase"
record_column = "irasas"

classes = ["N", "L", "R", "V", "A"]

# One fixed colour per class, used in every figure of this script.
class_colours = {
    "N": "#2a78d6",
    "L": "#eb6834",
    "R": "#1baf7a",
    "V": "#eda100",
    "A": "#e87ba4",
}

numeric_features = [
    "RR_pre", "RR_post", "RR_vid", "RR_sant_vid", "RR_sant_post",
    "PR", "QRS", "QT", "QTc",
    "P_amp", "T_amp", "ST", "QRS_max", "QRS_min",
]

# The record suspected of producing the most extreme RR values.
suspected_record = "232"
zoom_limits = (200, 1600)


def iqr_bounds(values):
    values = values.dropna()
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    return q1, q3, iqr, q1 - 1.5 * iqr, q3 + 1.5 * iqr


def outlier_percent(values):
    _, _, _, lower_bound, upper_bound = iqr_bounds(values)
    values = values.dropna()
    return ((values < lower_bound) | (values > upper_bound)).mean() * 100


def style_axis(axis):
    axis.grid(alpha=0.25, linewidth=0.6)
    axis.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Figure 1. RR_pre and RR_post by class.
# A premature beat comes early (short RR_pre) and is followed by a
# compensatory pause (long RR_post). If the V and A beats form their own
# cluster, the extreme RR values are physiological, not calculation errors.
# ---------------------------------------------------------------------------
def plot_rr_scatter(beats):
    figure, (left_axis, right_axis) = plt.subplots(1, 2, figsize=(12, 5.5))

    for axis, limits, title in [
        (left_axis, None, "Visos reikšmės"),
        (right_axis, zoom_limits, "Priartinta pagrindinė duomenų masė"),
    ]:
        # N, L and R are drawn first so the rarer V and A beats stay visible.
        for class_name in classes:
            class_beats = beats[beats[class_column] == class_name]
            axis.scatter(
                class_beats["RR_pre"], class_beats["RR_post"],
                s=10, alpha=0.45, linewidths=0,
                color=class_colours[class_name], label=class_name,
            )

        if limits:
            axis.set_xlim(*limits)
            axis.set_ylim(*limits)

        # RR_pre = RR_post means a regular rhythm.
        axis.plot(axis.get_xlim(), axis.get_xlim(),
                  color="#8a8a84", linewidth=1, linestyle="--", zorder=0)

        axis.set_xlabel("RR_pre, ms")
        axis.set_ylabel("RR_post, ms")
        axis.set_title(title)
        style_axis(axis)

    # Legend markers are larger and opaque so the colours read clearly.
    markers = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=8,
                   color=class_colours[class_name], label=class_name)
        for class_name in classes
    ]
    left_axis.legend(handles=markers, title="Dūžio klasė", loc="upper right")

    figure.tight_layout()
    figure.savefig(output_folder / "rr_sklaida.png", dpi=200)
    plt.close(figure)
    print("  ->", output_folder / "rr_sklaida.png")


def print_rr_medians(beats):
    medians = (
        beats
        .groupby(class_column)[["RR_pre", "RR_post"]]
        .median()
        .round(1)
        .reindex(classes)
    )
    medians["skirtumas"] = (medians["RR_post"] - medians["RR_pre"]).round(1)

    print("\nRR medianos pagal klasę (ms):")
    print(medians.to_string())
    medians.to_csv(output_folder / "rr_medianos.csv", encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Figure 2. Share of outliers per feature and class.
# The same numbers as in the table, but the pattern is visible at once.
# ---------------------------------------------------------------------------
def plot_outlier_heatmap(beats):
    shares = pd.DataFrame({
        class_name: [
            outlier_percent(beats.loc[beats[class_column] == class_name, feature])
            for feature in numeric_features
        ]
        for class_name in classes
    }, index=numeric_features)

    figure, axis = plt.subplots(figsize=(7.5, 7))

    # One hue from light to dark: a larger share means a darker cell.
    image = axis.imshow(shares.to_numpy(), cmap="Blues", aspect="auto", vmin=0)

    axis.set_xticks(range(len(classes)), classes)
    axis.set_yticks(range(len(numeric_features)), numeric_features)
    axis.set_xlabel("Dūžio klasė")
    axis.set_title("Išskirčių dalis pagal 1,5 × IQR taisyklę, %")

    # The value is printed in every cell so colour is not the only code.
    dark_cell_threshold = np.nanmax(shares.to_numpy()) * 0.6
    for row in range(shares.shape[0]):
        for column in range(shares.shape[1]):
            share = shares.iat[row, column]
            if pd.notna(share):
                axis.text(
                    column, row, f"{share:.1f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if share > dark_cell_threshold else "#0b0b0b",
                )

    figure.colorbar(image, ax=axis, label="Išskirčių dalis, %", shrink=0.8)
    figure.tight_layout()
    figure.savefig(output_folder / "isskirciu_dalys.png", dpi=200)
    plt.close(figure)
    print("  ->", output_folder / "isskirciu_dalys.png")

    shares.round(1).to_csv(output_folder / "isskirciu_dalys.csv", encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Figure 3. Does one ECG record produce the most extreme RR_pre values?
# ---------------------------------------------------------------------------
def plot_record_histogram(beats):
    class_r = beats[beats[class_column] == "R"].copy()
    class_r["is_suspected"] = class_r[record_column].astype(str) == suspected_record

    if not class_r["is_suspected"].any():
        print(f"\n{suspected_record} įrašo R klasėje nėra – 3 paveikslas nebraižomas.")
        return

    _, _, _, _, upper_bound = iqr_bounds(class_r["RR_pre"])

    figure, axis = plt.subplots(figsize=(10, 5))
    bin_edges = np.histogram_bin_edges(class_r["RR_pre"].dropna(), bins=60)

    axis.hist(
        [class_r.loc[~class_r["is_suspected"], "RR_pre"],
         class_r.loc[class_r["is_suspected"], "RR_pre"]],
        bins=bin_edges, stacked=True,
        color=["#1baf7a", "#4a3aa7"],
        label=["Kiti įrašai", f"{suspected_record} įrašas"],
    )

    axis.axvline(upper_bound, color="#0b0b0b", linewidth=1.5, linestyle="--")
    axis.annotate(
        f"IQR viršutinė riba\n{upper_bound:.0f} ms",
        xy=(upper_bound, axis.get_ylim()[1] * 0.75),
        xytext=(12, 0), textcoords="offset points", fontsize=9,
    )

    axis.set_xlabel("RR_pre, ms")
    axis.set_ylabel("Dūžių skaičius")
    axis.set_title("R klasės RR_pre pasiskirstymas pagal EKG įrašą")
    axis.legend()
    style_axis(axis)

    figure.tight_layout()
    figure.savefig(output_folder / "r_klases_rr_pre.png", dpi=200)
    plt.close(figure)
    print("  ->", output_folder / "r_klases_rr_pre.png")

    above_bound = class_r[class_r["RR_pre"] > upper_bound]
    print(f"\nR klasė, RR_pre virš IQR ribos ({upper_bound:.0f} ms): "
          f"{len(above_bound)} dūžiai")

    if len(above_bound):
        counts = above_bound[record_column].value_counts()
        print(counts.to_frame("duziu").to_string())
        share = (above_bound[record_column].astype(str) == suspected_record).mean() * 100
        print(f"{suspected_record} įrašo dalis tarp jų: {share:.1f} %")


def main():
    output_folder.mkdir(parents=True, exist_ok=True)
    beats = pd.read_csv(cleaned_file)

    plot_rr_scatter(beats)
    print_rr_medians(beats)
    plot_outlier_heatmap(beats)
    plot_record_histogram(beats)


if __name__ == "__main__":
    main()
