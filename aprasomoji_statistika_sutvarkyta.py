from pathlib import Path
import pandas as pd

CLEANED_DATA_FILE = Path("duomenys/tiriamoji_sutvarkyta.csv")
OVERALL_STATS_FILE = Path("duomenys/aprasomoji_statistika_sutvarkyta.csv")
CLASS_STATS_DIR = Path("duomenys/apras_stat_klases_sutvarkyta")

CLASS_COLUMN = "klase"
CLASSES = ["N", "L", "R", "V", "A"]

NUMERIC_FEATURES = [
    "RR_pre",
    "RR_post",
    "RR_vid",
    "RR_sant_vid",
    "RR_sant_post",
    "PR",
    "QRS",
    "QT",
    "QTc",
    "P_amp",
    "T_amp",
    "ST",
    "QRS_max",
    "QRS_min",
]

# Kategoriniams požymiams skaičiuojame tik trūkstamas reikšmes
CATEGORICAL_FEATURES = [
    "P_yra",
    "T_tipas",
]

MISSING_COLUMN = "Neegzistuojančios reikšmės"
NOT_APPLICABLE = "–"
DECIMALS = 4


def format_number(value):
    text = f"{value:.{DECIMALS}f}".rstrip("0").rstrip(".")

    if text == "-0":
        return "0"

    return text


def numeric_feature_stats(data):
    numeric_values = data[NUMERIC_FEATURES]

    stats = pd.DataFrame({
        "Minimumas": numeric_values.min(),
        "1-as kvartilis": numeric_values.quantile(0.25),
        "Mediana": numeric_values.median(),
        "Vidurkis": numeric_values.mean(),
        "3-as kvartilis": numeric_values.quantile(0.75),
        "Maksimumas": numeric_values.max(),
        "Dispersija": numeric_values.var(),
    }).map(format_number)

    stats[MISSING_COLUMN] = numeric_values.isna().sum()

    return stats


def categorical_feature_stats(data, columns):
    stats = pd.DataFrame(
        NOT_APPLICABLE,
        index=CATEGORICAL_FEATURES,
        columns=columns,
    )

    stats[MISSING_COLUMN] = data[CATEGORICAL_FEATURES].isna().sum()

    return stats


def descriptive_stats_table(data):
    numeric_stats = numeric_feature_stats(data)
    categorical_stats = categorical_feature_stats(data, numeric_stats.columns)

    table = pd.concat([numeric_stats, categorical_stats])
    table.index.name = "Požymis"

    return table


def print_and_save_stats(data, title, output_file):
    stats_table = descriptive_stats_table(data)

    print(f"\n{title} (n = {len(data)})\n")
    print(stats_table.to_string())

    stats_table.to_csv(output_file, encoding="utf-8-sig")


if __name__ == "__main__":
    cleaned_data = pd.read_csv(CLEANED_DATA_FILE)
    CLASS_STATS_DIR.mkdir(parents=True, exist_ok=True)

    print_and_save_stats(
        cleaned_data,
        "Visa sutvarkyta aibė",
        OVERALL_STATS_FILE,
    )

    for class_name in CLASSES:
        class_data = cleaned_data[cleaned_data[CLASS_COLUMN] == class_name]

        print_and_save_stats(
            class_data,
            f"Klasė {class_name}",
            CLASS_STATS_DIR / f"{class_name}_klase_AP.csv",
        )
