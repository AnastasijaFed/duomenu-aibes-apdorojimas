import pandas as pd
from scipy.stats import kruskal

DATA_FILE = "duomenys/tiriamoji_sutvarkyta.csv"
CLASS_COLUMN = "klase"
CLASSES = ["N", "L", "R", "V", "A"]
SIGNIFICANCE_LEVEL = 0.05

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
    "P_yra",
    "T_tipas",
]


def add_rr_ratios(df):
    df = df.copy()

    df["RR_sant_vid"] = df["RR_pre"] / df["RR_vid"]
    df["RR_sant_post"] = df["RR_pre"] / df["RR_post"]

    return df


def kruskal_wallis_table(df, features, class_column, classes):
    rows = []
    for feature in features:
        values_by_class = []

        for class_name in classes:
            rows_of_this_class = df[df[class_column] == class_name]
            feature_values = rows_of_this_class[feature]
            feature_values_without_missing = feature_values.dropna()

            values_by_class.append(feature_values_without_missing)

        h_statistic, p_value = kruskal(*values_by_class)

        total_count = sum(len(values) for values in values_by_class)

        epsilon_squared = h_statistic / (total_count - 1)

        rows.append({
            "feature": feature,
            "n": total_count,
            "H": h_statistic,
            "p_value": p_value,
            "epsilon_sq": epsilon_squared,
            "significant": p_value < SIGNIFICANCE_LEVEL,
        })

    results = pd.DataFrame(rows)

    return results.sort_values("H", ascending=False).reset_index(drop=True)


def format_p_value(p_value):
    if p_value < 0.001:
        return "< 0.001"

    return f"{p_value:.3f}"


if __name__ == "__main__":
    data = pd.read_csv(DATA_FILE)
    data = add_rr_ratios(data)

    results = kruskal_wallis_table(data, FEATURES, CLASS_COLUMN, CLASSES)

    print(results.to_string(
        index=False,
        formatters={
            "H": "{:.2f}".format,
            "p_value": format_p_value,
            "epsilon_sq": "{:.3f}".format,
        },
    ))
