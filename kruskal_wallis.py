import pandas as pd
from scipy.stats import kruskal, chi2_contingency

DATA_FILE = "duomenys/tiriamoji_sutvarkyta.csv"
RESULTS_NUMERIC_FILE = "duomenys/kruskal_wallis.csv"
RESULTS_CATEGORICAL_FILE = "duomenys/chi_kvadratas.csv"

CLASS_COLUMN = "klase"
CLASSES = ["N", "L", "R", "V", "A"]

# The same level as in the correlation section of the report.
SIGNIFICANCE_LEVEL = 0.01

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

# P_yra is binary and T_tipas is a nominal code without ordering,
# so neither may be ranked.
CATEGORICAL_FEATURES = [
    "P_yra",
    "T_tipas",
]


def effect_size_label(effect_size):
    if effect_size < 0.01:
        return "labai nezymus"
    if effect_size < 0.06:
        return "nedidelis"
    if effect_size < 0.14:
        return "vidutinis"
    return "stiprus"


def kruskal_wallis_table(data):
    rows = []

    for feature in NUMERIC_FEATURES:
        values_by_class = [
            data.loc[data[CLASS_COLUMN] == class_name, feature].dropna()
            for class_name in CLASSES
        ]

        h_statistic, p_value = kruskal(*values_by_class)

        class_count = len(values_by_class)
        total_count = sum(len(values) for values in values_by_class)

        # Two effect size measures for the same test.
        epsilon_squared = h_statistic / (total_count - 1)
        eta_squared = (h_statistic - class_count + 1) / (total_count - class_count)

        rows.append({
            "pozymis": feature,
            "n": total_count,
            "H": round(h_statistic, 1),
            "p": p_value,
            "epsilon_sq": round(epsilon_squared, 3),
            "eta_sq": round(eta_squared, 3),
            "efektas": effect_size_label(epsilon_squared),
            "reiksminga": p_value < SIGNIFICANCE_LEVEL,
        })

    results = pd.DataFrame(rows)
    return results.sort_values("epsilon_sq", ascending=False).reset_index(drop=True)


def chi_square_table(data):
    rows = []

    for feature in CATEGORICAL_FEATURES:
        counts = pd.crosstab(data[CLASS_COLUMN], data[feature])
        chi_square, p_value, degrees_of_freedom, expected = chi2_contingency(counts)

        total_count = int(counts.to_numpy().sum())
        smaller_dimension = min(counts.shape) - 1
        cramers_v = (chi_square / (total_count * smaller_dimension)) ** 0.5

        rows.append({
            "pozymis": feature,
            "n": total_count,
            "chi2": round(float(chi_square), 1),
            "laisves_laipsniai": int(degrees_of_freedom),
            "p": float(p_value),
            "Cramer_V": round(float(cramers_v), 3),
            "reiksminga": p_value < SIGNIFICANCE_LEVEL,
        })

    return pd.DataFrame(rows)


def format_p_value(p_value):
    if p_value < 0.001:
        return "< 0,001"
    return f"{p_value:.3f}".replace(".", ",")


def main():
    data = pd.read_csv(DATA_FILE)

    numeric_results = kruskal_wallis_table(data)
    print("KRUSKAL-WALLIS (kiekybiniai pozymiai):")
    print(numeric_results.to_string(
        index=False,
        formatters={"p": format_p_value},
    ))
    numeric_results.to_csv(RESULTS_NUMERIC_FILE, index=False, encoding="utf-8-sig")

    categorical_results = chi_square_table(data)
    print("\nCHI KVADRATAS (kategoriniai pozymiai):")
    print(categorical_results.to_string(
        index=False,
        formatters={"p": format_p_value},
    ))
    categorical_results.to_csv(RESULTS_CATEGORICAL_FILE, index=False, encoding="utf-8-sig")

    print("\nStipriausiai klases atskiriantys kiekybiniai pozymiai:")
    print(numeric_results.head(6)[["pozymis", "epsilon_sq", "efektas"]].to_string(index=False))


if __name__ == "__main__":
    main()