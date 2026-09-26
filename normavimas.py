from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

input_file = "duomenys/tiriamoji_sutvarkyta.csv"
output_folder = Path("duomenys/normavimas")

class_column = "klase"
classes = ["N", "L", "R", "V", "A"]

numeric_features = [
    "RR_pre", "RR_post", "RR_vid", "RR_sant_vid", "RR_sant_post",
    "PR", "QRS", "QT", "QTc",
    "P_amp", "T_amp", "ST", "QRS_max", "QRS_min",
]
title_size = 16
label_size = 14        
tick_size = 13        
legend_size = 13

def min_max(x):
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min)


def mean_and_variance(x):
    mean = x.mean()
    variance = x.var()
    return (x - mean) / variance ** 0.5


def robust_scalling(x):
    median = x.median()
    iqr = x.quantile(0.75) - x.quantile(0.25)
    return (x - median) / iqr


methods = {
    "min_max": ("", min_max),
    "vid_ir_disp": ("", mean_and_variance),
    "atsparusis_mastelio_keitimas": ("", robust_scalling),
}


def normalize(data, method):
    normalized = data.copy()

    for feature in numeric_features:
        normalized[feature] = method(data[feature])

    return normalized


def medians_by_class(data):
    return data.groupby(class_column)[numeric_features].median().T[classes]



#visualization
feature_colours = dict(zip(numeric_features, plt.get_cmap("tab20").colors))
features_to_show = 6


def features_that_differ_most(medians):
    spread = medians.max(axis=1) - medians.min(axis=1)
    top_features = spread.sort_values(ascending=False).index[:features_to_show]
    return [feature for feature in numeric_features if feature in top_features]


def plot_medians_bar_chart(medians, features, title, y_label, file_name):
    figure, axis = plt.subplots(figsize=(13, 6))
    bar_width = 0.9 / len(features)
    positions = np.arange(len(classes))

    for number, feature in enumerate(features):
        axis.bar(positions + number * bar_width, medians.loc[feature, classes],
                 width=bar_width, color=feature_colours[feature], label=feature)

    axis.set_facecolor("white")
    axis.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    axis.set_axisbelow(True)
    for side in axis.spines.values():
        side.set_visible(False)
    axis.tick_params(length=0)
    axis.axhline(0, color="#9e9e9e", linewidth=0.8)
    axis.set_xticks(positions + 0.9 / 2 - bar_width / 2, classes, fontsize=tick_size + 2)
    axis.tick_params(axis="y", labelsize=tick_size)
    axis.set_xlabel("Dūžio klasė", fontsize=label_size)
    axis.set_ylabel(y_label, fontsize=label_size)
    axis.set_title(title, loc="center", fontsize=title_size)
    axis.legend(title="Požymis", bbox_to_anchor=(1.01, 0.5), loc="center left",
                frameon=False, fontsize=legend_size, title_fontsize=legend_size + 1)

    save_figure(file_name)


def plot_one_class(medians, class_name, title, y_limits, file_path):
    figure, axis = plt.subplots(figsize=(10, 5))
    colours = [feature_colours[feature] for feature in numeric_features]
    axis.bar(numeric_features, medians[class_name], color=colours, width=0.8)
    axis.set_facecolor("white")
    axis.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    axis.set_axisbelow(True)
    for side in axis.spines.values():
        side.set_visible(False)
    axis.tick_params(length=0)

    axis.axhline(0, color="#bdbdbd", linewidth=0.8)
    axis.set_ylim(y_limits)
    axis.set_title(title, loc="left", fontsize=title_size)
    axis.set_xlabel("Požymis", fontsize=label_size)
    axis.set_ylabel("Normuota reikšmė (mediana)", fontsize=label_size)
    axis.tick_params(axis="y", labelsize=tick_size)
    plt.xticks(rotation=35, ha="right", fontsize=tick_size)

    plt.tight_layout()
    plt.savefig(file_path, dpi=200)
    plt.close()


def plot_every_class_separately(medians, method_name):
    margin = 0.1 * (medians.max().max() - medians.min().min())
    y_limits = (min(0, medians.min().min()) - margin, max(0, medians.max().max()) + margin)

    class_folder = output_folder / "klases" / method_name
    class_folder.mkdir(parents=True, exist_ok=True)

    for class_name in classes:
        title = f"Normuotos medianos: {class_name} klasė ({method_name})"
        plot_one_class(medians, class_name, title, y_limits, class_folder / f"{class_name}_klase.png")


def save_figure(file_name):
    plt.tight_layout()
    plt.savefig(output_folder / file_name, dpi=200)
    plt.close()

def print_min_max_squeeze(min_max_data):
    values = min_max_data[numeric_features]
    middle_half = values.quantile(0.75) - values.quantile(0.25)

    print("\nMin–max: vidurinės 50 % reikšmių užimama intervalo [0; 1] dalis:")
    print(middle_half.round(3).sort_values())


def main():
    output_folder.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(input_file)
    plot_medians_bar_chart(
        medians_by_class(data),
        numeric_features,
        "Nenormuotų požymių medianos pagal dūžio klasę",
        "Mediana",
        "nenormuotos_medianos.png",
    )

    normalized_data = {}
    medians = {}

    for method_name, (title, method) in methods.items():
        normalized = normalize(data, method)
        normalized.to_csv(output_folder / f"tiriamoji_{method_name}.csv", index=False)
        normalized_data[method_name] = normalized

        medians[method_name] = medians_by_class(normalized)
        print(f"\n{title} – požymių medianos pagal klasę:")
        print(medians[method_name].round(3))

    shown_features = features_that_differ_most(medians["atsparusis_mastelio_keitimas"])
    print("\nDiagramose rodomi požymiai:", shown_features)

    for method_name, (title, method) in methods.items():
        plot_medians_bar_chart(
            medians[method_name],
            shown_features,
            f"Normuotos medianos kiekvienai dūžio klasei",
            "Normuota reikšmė (mediana)",
            f"medianos_{method_name}.png",
        )
        plot_every_class_separately(medians[method_name], method_name)

    print_min_max_squeeze(normalized_data["min_max"])


if __name__ == "__main__":
    main()