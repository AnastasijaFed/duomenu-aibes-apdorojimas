from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

input_file = "duomenys/tiriamoji.csv"
output_folder = Path("duomenys/aprasomoji_vizualizacija")

class_column = "klase"
classes = ["N", "L", "R", "V", "A"]

features_with_anomalies = {
    "RR_pre": "ms",
    "RR_post": "ms",
    "RR_vid": "ms",
    "RR_sant_post": "",
    "PR": "ms",
    "QRS": "ms",
    "QTc": "ms",
}

amplitude_features = ["P_amp", "T_amp", "ST", "QRS_max", "QRS_min"]

outlier_colour = "#f5a623"
extreme_colour = "#8b0000"

title_size = 16
label_size = 14
tick_size = 13
legend_size = 12


def outlier_types(values):
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    outside_1_5 = (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)
    outside_3 = (values < q1 - 3 * iqr) | (values > q3 + 3 * iqr)

    return outside_1_5 & ~outside_3, outside_3


def count_outliers(data, features):
    rows = []

    for feature in features:
        for class_name in classes:
            values = data.loc[data[class_column] == class_name, feature].dropna()
            is_outlier, is_extreme = outlier_types(values)

            rows.append({
                "pozymis": feature,
                "klase": class_name,
                "reiksmiu": len(values),
                "isskirtys": is_outlier.sum(),
                "ekstremalios": is_extreme.sum(),
            })

    return pd.DataFrame(rows)

def draw_class_boxplots(axis, data, feature):
    positions = range(len(classes), 0, -1)

    for position, class_name in zip(positions, classes):
        values = data.loc[data[class_column] == class_name, feature].dropna()

        axis.boxplot(values, positions=[position], vert=False, widths=0.6,
                     showfliers=False, patch_artist=True,
                     boxprops={"facecolor": "#d9d9d9"},
                     medianprops={"color": "black", "linewidth": 2})

        is_outlier, is_extreme = outlier_types(values)
        axis.scatter(values[is_outlier], [position] * is_outlier.sum(),
                     s=18, color=outlier_colour, zorder=3)
        axis.scatter(values[is_extreme], [position] * is_extreme.sum(),
                     s=18, color=extreme_colour, zorder=3)

    axis.set_yticks(list(positions), classes, fontsize=tick_size)
    axis.tick_params(axis="x", labelsize=tick_size - 1)
    axis.grid(axis="x", color="#e6e6e6", linewidth=0.8)
    axis.set_axisbelow(True)


def add_outlier_legend(figure_or_axis):
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=outlier_colour,
                   label="Išskirtis (1,5–3 IQR)"),
        plt.Line2D([], [], marker="o", linestyle="", color=extreme_colour,
                   label="Ekstremali išskirtis (> 3 IQR)"),
    ]
    figure_or_axis.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1),
                          fontsize=legend_size, frameon=False)


def plot_feature(data, feature, unit):
    figure, axis = plt.subplots(figsize=(13, 5.5))

    draw_class_boxplots(axis, data, feature)

    axis_name = f"{feature}, {unit}" if unit else feature
    axis.set_xlabel(axis_name, fontsize=label_size)
    axis.set_ylabel("Dūžio klasė", fontsize=label_size)
    axis.set_title(f"Požymio {feature} stačiakampės diagramos pagal klasę",
                   loc="center", fontsize=title_size)
    add_outlier_legend(axis)

    plt.tight_layout()
    plt.savefig(output_folder / f"staciakampe_{feature}.png", dpi=200)
    plt.close()


def plot_amplitudes(data):
    figure, axes = plt.subplots(1, len(amplitude_features), figsize=(20, 6), sharey=True)

    for axis, feature in zip(axes, amplitude_features):
        draw_class_boxplots(axis, data, feature)
        axis.axvline(0, color="#9e9e9e", linewidth=0.8)
        axis.set_title(feature, fontsize=label_size)
        axis.set_xlabel("signalo vnt.", fontsize=label_size - 1)

    axes[0].set_ylabel("Dūžio klasė", fontsize=label_size)
    figure.suptitle("Amplitudžių požymių stačiakampės diagramos pagal klasę", fontsize=title_size)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=outlier_colour,
                   label="Išskirtis (1,5–3 IQR)"),
        plt.Line2D([], [], marker="o", linestyle="", color=extreme_colour,
                   label="Ekstremali išskirtis (> 3 IQR)"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=2, fontsize=legend_size, frameon=False)

    plt.tight_layout(rect=(0, 0.06, 1, 1))
    plt.savefig(output_folder / "staciakampes_amplitudes.png", dpi=200)
    plt.close()

def main():
    output_folder.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(input_file)

    for feature, unit in features_with_anomalies.items():
        plot_feature(data, feature, unit)

    plot_amplitudes(data)
    counts = count_outliers(data, list(features_with_anomalies) + amplitude_features)
    counts.to_csv(output_folder / "isskirciu_kiekiai.csv", index=False)

    print("Išskirčių ir ekstremalių išskirčių kiekiai pagal klasę:")
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()