from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

DATA_FILE = Path("duomenys/tiriamoji_sutvarkyta.csv")
OUTPUT_DIR = Path("duomenys/koreliacija")
CLASS_COLUMN = "klase"
CLASSES = ["N", "L", "R", "V", "A"]
ALL_DATA_NAME = "visa_aibe"

# Medicininiams duomenims naudojame griežtesnę reikšmingumo ribą
SIGNIFICANCE_LEVEL = 0.01

# Kategoriniai požymiai (P_yra, T_tipas) neįtraukiami,
# nes T_tipas kodai neturi eiliškumo.
FEATURES = [
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


def correlation_strength(rho):
    abs_rho = abs(rho)

    if abs_rho >= 0.7:
        return "stipri"

    if abs_rho >= 0.3:
        return "vidutinė"

    return "silpna"


def spearman_correlation_table(df, features):
    rows = []

    for feature_1, feature_2 in combinations(features, 2):
        # Trūkstamas reikšmes šaliname poromis: naudojame tik tas eilutes,
        # kuriose yra abiejų požymių reikšmės.
        pair_values = df[[feature_1, feature_2]].dropna()

        rho, p_value = spearmanr(pair_values[feature_1], pair_values[feature_2])

        rows.append({
            "feature_1": feature_1,
            "feature_2": feature_2,
            "n": len(pair_values),
            "rho": rho,
            "p_value": p_value,
            "strength": correlation_strength(rho),
            "significant": p_value < SIGNIFICANCE_LEVEL,
        })

    results = pd.DataFrame(rows)

    results["abs_rho"] = results["rho"].abs()
    results = results.sort_values("abs_rho", ascending=False)

    return results.drop(columns="abs_rho").reset_index(drop=True)


def rho_by_data_set(tables_by_data_set):
    # Viena lentelė su kiekvienos poros rho visoje aibėje ir kiekvienoje klasėje,
    # kad būtų lengva matyti, kaip ryšys keičiasi tarp klasių.
    rho_columns = []

    for data_set_name, table in tables_by_data_set.items():
        rho_column = (
            table
            .set_index(["feature_1", "feature_2"])["rho"]
            .rename(data_set_name)
        )
        rho_columns.append(rho_column)

    comparison = pd.concat(rho_columns, axis=1)

    class_rho = comparison[CLASSES]
    comparison["range_between_classes"] = class_rho.max(axis=1) - class_rho.min(axis=1)

    return (
        comparison
        .sort_values("range_between_classes", ascending=False)
        .reset_index()
    )


def format_p_value(p_value):
    if p_value < 0.001:
        return "< 0.001"

    return f"{p_value:.3f}"


def format_correlation_table(table):
    formatted = table.copy()

    formatted["rho"] = formatted["rho"].round(4)
    formatted["p_value"] = formatted["p_value"].map(format_p_value)

    return formatted


if __name__ == "__main__":
    data = pd.read_csv(DATA_FILE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data_sets = {ALL_DATA_NAME: data}

    for class_name in CLASSES:
        data_sets[class_name] = data[data[CLASS_COLUMN] == class_name]

    tables_by_data_set = {}

    for data_set_name, data_set in data_sets.items():
        table = spearman_correlation_table(data_set, FEATURES)
        tables_by_data_set[data_set_name] = table

        formatted_table = format_correlation_table(table)

        print(f"\n{data_set_name} (n = {len(data_set)})\n")
        print(formatted_table.to_string(index=False))

        formatted_table.to_csv(
            OUTPUT_DIR / f"{data_set_name}_koreliacija.csv",
            index=False,
            encoding="utf-8-sig",
        )

    comparison = rho_by_data_set(tables_by_data_set).round(4)

    print("\nRHO PALYGINIMAS TARP KLASIŲ:\n")
    print(comparison.to_string(index=False))

    comparison.to_csv(
        OUTPUT_DIR / "rho_palyginimas_tarp_klasiu.csv",
        index=False,
        encoding="utf-8-sig",
    )
