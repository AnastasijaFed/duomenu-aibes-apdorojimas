# Missing values, duplicates, invalid values, outliers, hypotheses about outliers and problematic features.

import pandas as pd

all_beats_file = "duomenys/duziai.csv"
sample_file = "duomenys/tiriamoji.csv"
complete_rows_file = "duomenys/tiriamoji_analizei.csv"

class_column = "klase"
record_column = "irasas"
rhythm_column = "ritmas"
matched_column = "rastas"
p_wave_column = "P_yra"
t_wave_type_column = "T_tipas"

classes = ["N", "L", "R", "V", "A"]
excluded_record = "114"

numeric_features = [
    "RR_pre", "RR_post", "RR_vid", "RR_sant_vid", "RR_sant_post",
    "PR", "QRS", "QT", "QTc",
    "P_amp", "T_amp", "ST", "QRS_max", "QRS_min",
]
categorical_features = [p_wave_column, t_wave_type_column]
all_features = numeric_features + categorical_features

interval_features = [
    "RR_pre", "RR_post", "RR_vid", "RR_sant_vid", "RR_sant_post",
    "PR", "QRS", "QT", "QTc",
]

control_limits = {
    "PR": (50, 400),
    "QRS": (40, 250),
    "QT": (200, 700),
}

candidate_numeric = ["QRS", "RR_sant_vid", "RR_sant_post", "RR_pre", "QTc"]


def print_missing_values(sample):
    features = sample[all_features]

    missing = pd.DataFrame({
        "count": features.isna().sum(),
        "percent": features.isna().mean() * 100,
    })

    print("Trūkstamos reikšmės:")
    print("\nBendrai:")
    print(missing.round(2))

    missing_by_class = (
        sample
        .groupby(class_column)[all_features]
        .apply(lambda class_beats: class_beats.isna().mean() * 100)
    )

    print("\nPagal klases:")
    print(missing_by_class.round(2))


def print_ecgpuwave_matching(sample):
    matched_by_class = sample.groupby(class_column)[matched_column].agg(["sum", "count", "mean"])
    matched_by_class["percent"] = matched_by_class["mean"] * 100

    print("\nECGPUWAVE dūžių suderinimas:")
    print(matched_by_class[["sum", "count", "percent"]].round(2))


def print_p_wave_by_class(sample):
    p_wave_share = pd.crosstab(
        sample[class_column],
        sample[p_wave_column].fillna(-1),
        normalize="index",
    ) * 100

    print("\nP banga pagal klasę (%):")
    print("-1 = dūžis nesuderintas, 0 = P banga neaptikta, 1 = P banga aptikta")
    print(p_wave_share.round(1))


def print_missing_by_matching(sample):
    print("\nTrūkstamos reikšmės pagal 'rastas':")

    for feature in ["PR", "QRS", "QT", "P_amp", "T_amp", t_wave_type_column]:
        print(f"\n{feature}:")
        print(pd.crosstab(sample[matched_column], sample[feature].isna()))


def share_without_p_wave(beats, group_by):
    table = (
        (beats[p_wave_column] == 0)
        .groupby(beats[group_by], dropna=False)
        .agg(["mean", "size"])
    )
    table["mean"] = (table["mean"] * 100).round(1)
    return table.rename(columns={"mean": "P_missing_%", "size": "beats"})


def print_normal_beats_without_p_wave(study_beats):
    normal_beats = study_beats[
        (study_beats[class_column] == "N") & study_beats[p_wave_column].notna()
    ].copy()

    print("\nN dūžiai be P bangos pagal ritmą:")
    by_rhythm = share_without_p_wave(normal_beats, rhythm_column)
    print(by_rhythm.sort_values("beats", ascending=False))

    sinus_beats = normal_beats[normal_beats[rhythm_column] == "(N"].copy()

    print("\nP nerasta N dūžiuose sinusiniame ritme pagal įrašą:")
    by_record = share_without_p_wave(sinus_beats, record_column)
    print(by_record.sort_values("P_missing_%"))



def print_duplicates(sample):
    duplicate_count = sample.duplicated(subset=[record_column, "R"]).sum()
    print("Dublikatų (irasas + R):", duplicate_count)


def print_non_positive_values(sample):
    print("\nNelogiškos reikšmės (<= 0):")

    for feature in interval_features:
        print(feature, (sample[feature] <= 0).sum())


def print_values_outside_limits(sample):
    print("\nUž kontrolinių ribų pagal klasę:")

    for feature, (low, high) in control_limits.items():
        is_outside = sample[feature].notna() & ~sample[feature].between(low, high)

        print(f"\n{feature} ({low}–{high} ms):")
        print(is_outside.groupby(sample[class_column]).agg(["sum", "count"]))

    print("\nUž ribų (žemiau/aukščiau):")

    for feature, (low, high) in control_limits.items():
        values = sample[feature]
        print(
            f"{feature}: "
            f"< {low}: {(values < low).sum()}, "
            f"> {high}: {(values > high).sum()}, "
            f"min = {values.min():.1f}, "
            f"max = {values.max():.1f}"
        )


def find_outside_limits(sample, feature, low, high, columns):
    is_outside = sample[feature].notna() & ~sample[feature].between(low, high)
    return sample.loc[is_outside, columns]


def print_suspicious_qrs_and_qt(sample):
    suspicious_qrs = find_outside_limits(
        sample, "QRS", 40, 250,
        [record_column, "R", class_column, "QRS", "QRS_on", "QRS_off"],
    )
    suspicious_qt = find_outside_limits(
        sample, "QT", 200, 700,
        [record_column, "R", class_column, "QT", "QRS_on", "T_off"],
    )

    print("\nĮtartinos QRS reikšmės:")
    print(suspicious_qrs.sort_values("QRS").to_string(index=False))

    print("\nĮtartinos QT reikšmės:")
    print(suspicious_qt.sort_values("QT").to_string(index=False))

    print("\nQRS už ribų pagal įrašą:")
    print(suspicious_qrs.groupby([record_column, class_column]).size().sort_values(ascending=False))

    print("\nQT už ribų pagal įrašą:")
    print(suspicious_qt.groupby([record_column, class_column]).size().sort_values(ascending=False))


def iqr_outlier_percent(values):
    #values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    values = values.dropna()#missing values are not outliers
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    is_outlier = (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)
    return is_outlier.mean() * 100


def print_iqr_outliers(sample):
    outliers_by_class = (
        sample
        .groupby(class_column)[numeric_features]
        .agg(iqr_outlier_percent)
        .T
    )

    print("\nIšskirtys pagal IQR taisyklę klasės viduje:")
    print(outliers_by_class.round(1))


def print_rr_ratios_by_class(sample):
    print("\nAbiejų RR santykių palyginimas pagal klasę:")

    for feature in ["RR_sant_vid", "RR_sant_post"]:
        print(f"\n{feature}:")
        print(
            sample
            .groupby(class_column)[feature]
            .agg(
                count="count",
                mean="mean",
                median="median",
                std="std",
                min="min",
                max="max",
            )
            .round(3)
        )

def check_record_232_in_class_a(complete_rows):
    is_class_a = complete_rows[class_column] == "A"
    is_record_232 = complete_rows[record_column].astype(str) == "232"

    print("\nA klasė: 232 įrašas (True) vs kiti (False), medianos:")
    print(
        complete_rows[is_class_a]
        .groupby(is_record_232[is_class_a])[["QRS", "RR_post"]]
        .median()
        .round(1)
    )


def check_long_rr_pre_in_class_r(complete_rows):
    class_r = complete_rows[complete_rows[class_column] == "R"]
    long_rr_pre = class_r["RR_pre"] > 2000

    print("\nR klasės RR_pre > 2000 ms pagal įrašą:")
    print(class_r.loc[long_rr_pre, record_column].value_counts())


def check_afib_among_normal_outliers(complete_rows):
    class_n = complete_rows[complete_rows[class_column] == "N"]
    ratio = class_n["RR_sant_vid"]

    q1 = ratio.quantile(0.25)
    q3 = ratio.quantile(0.75)
    iqr = q3 - q1

    is_outlier = ~ratio.between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    is_afib = class_n[rhythm_column] == "(AFIB"

    print(
        f"\nN RR_sant_vid: AFIB dalis tarp išskirčių "
        f"{is_afib[is_outlier].mean() * 100:.1f} %, "
        f"tarp kitų {is_afib[~is_outlier].mean() * 100:.1f} %"
    )

def print_problematic_features(sample, complete_rows):
    print("\nMasteliai (min–max):")
    print(complete_rows[candidate_numeric].agg(["min", "max"]).round(2))

    print("\nT_tipas kiekiai (retos kategorijos):")
    print(pd.crosstab(complete_rows[class_column], complete_rows[t_wave_type_column]))

    print("\nAsimetrijos koeficientas (visa tiriamoji aibė):")
    print(
        sample[numeric_features]
        .skew()
        .round(4)
        .sort_values(ascending=False)
    )


def main():
    all_beats = pd.read_csv(all_beats_file)
    sample = pd.read_csv(sample_file)
    complete_rows = pd.read_csv(complete_rows_file)

    study_beats = all_beats[
        all_beats[class_column].isin(classes)
        & (all_beats[record_column].astype(str) != excluded_record)
    ].copy()

    print_missing_values(sample)
    print_ecgpuwave_matching(sample)
    print_p_wave_by_class(sample)
    print_missing_by_matching(sample)
    print_normal_beats_without_p_wave(study_beats)

    print_duplicates(sample)
    print_non_positive_values(sample)
    print_values_outside_limits(sample)
    print_suspicious_qrs_and_qt(sample)

    print_iqr_outliers(sample)
    print_rr_ratios_by_class(sample)

    check_record_232_in_class_a(complete_rows)
    check_long_rr_pre_in_class_r(complete_rows)
    check_afib_among_normal_outliers(complete_rows)

    print_problematic_features(sample, complete_rows)


if __name__ == "__main__":
    main()