#   1. Min-max:              x_norm   = (x - x_min) / (x_max - x_min)
#   2. Mean and variance:    x_norm   = (x - mean) / sqrt(variance)
#   3. Robust scaling:       x_robust = (x - Med(x)) / IQR,   IQR = Q3 - Q1
from pathlib import Path

import matplotlib.pyplot as plt
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

def min_max(x):
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min)


def mean_and_variance(x):
    mean = x.mean()
    variance = x.var()
    return (x - mean) / variance ** 0.5


def robust(x):
    median = x.median()
    iqr = x.quantile(0.75) - x.quantile(0.25)
    return (x - median) / iqr


methods = {
    "min_max": ("Min–max", min_max),
    "vid_ir_disp": ("vid_ir_disp", mean_and_variance),
    "atsparusis_mastelio_keitimas": ("atsparusis_mastelio_keitimas", robust),
}


def normalize(data, method):
    normalized = data.copy()

    for feature in numeric_features:
        normalized[feature] = method(data[feature])

    return normalized


def medians_by_class(data):
    return data.groupby(class_column)[numeric_features].median().T[classes]

feature_colours = plt.get_cmap("tab20").colors[:len(numeric_features)]


def plot_medians_bar_chart(medians, title, y_label, file_name):
    figure, axis = plt.subplots(figsize=(13, 6))

    medians.T.plot(kind="bar", ax=axis, width=0.9, color=feature_colours, rot=0)
    axis.set_facecolor("white")
    axis.grid(color="#e6e6e6", linewidth=0.8)
    axis.set_axisbelow(True)
    for side in axis.spines.values():
        side.set_visible(False)
    axis.tick_params(length=0)

    axis.axhline(0, color="#bdbdbd", linewidth=0.8)
    axis.set_title(title, loc="left", fontsize=13)
    axis.set_xlabel("Dūžio klasė")
    axis.set_ylabel(y_label)
    axis.legend(title="Požymis", bbox_to_anchor=(1.01, 0.5), loc="center left",
                frameon=False, fontsize=9)

    save_figure(file_name)


def plot_one_class(medians, class_name, title, y_limits, file_path):
    figure, axis = plt.subplots(figsize=(10, 5))

    axis.bar(numeric_features, medians[class_name], color=feature_colours, width=0.8)

    axis.set_facecolor("white")
    axis.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    axis.set_axisbelow(True)
    for side in axis.spines.values():
        side.set_visible(False)
    axis.tick_params(length=0)

    axis.axhline(0, color="#bdbdbd", linewidth=0.8)
    axis.set_ylim(y_limits)
    axis.set_title(title, loc="left", fontsize=13)
    axis.set_xlabel("Požymis")
    axis.set_ylabel("Normuota reikšmė (mediana)")
    plt.xticks(rotation=35, ha="right")

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
        "Nenormuotų požymių medianos pagal dūžio klasę",
        "Mediana",
        "nenormuotos_medianos.png",
    )

    normalized_data = {}

    for method_name, (title, method) in methods.items():
        normalized = normalize(data, method)
        normalized.to_csv(output_folder / f"tiriamoji_{method_name}.csv", index=False)
        normalized_data[method_name] = normalized

        medians = medians_by_class(normalized)
        print(f"\n{title} – požymių medianos pagal klasę:")
        print(medians.round(3))

        plot_medians_bar_chart(
            medians,
            f"Normuotos medianos pagal dūžio klasę ({title})",
            "Normuota reikšmė (mediana)",
            f"medianos_{method_name}.png",
        )
        plot_every_class_separately(medians, method_name)

    print_min_max_squeeze(normalized_data["min_max"])


if __name__ == "__main__":
    main()