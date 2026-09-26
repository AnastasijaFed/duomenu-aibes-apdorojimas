from pathlib import Path

import pandas as pd

DATA_FILE = Path("duomenys/tiriamoji.csv")
OUTPUT_DIR = Path("duomenys/isskirciu_patikra")

SAMPLING_RATE = 360
CLASSES = ["N", "L", "R", "V", "A"]
QRS_LIMITS = (40, 250)
QT_LIMITS = (200, 700)

QRS_COLUMNS = ["irasas", "R", "klase", "QRS", "QRS_on", "QRS_off", "T_on",
               "into_T_wave_ms", "past_next_R_ms", "amplitude_ratio", "verdict"]

QT_COLUMNS = ["irasas", "R", "klase", "QT", "QRS_on", "T_off",
              "RR_post", "past_next_R_ms", "QT_to_RR_ratio", "verdict"]


def samples_to_ms(samples):
    return samples / SAMPLING_RATE * 1000


def save(table, file_name):
    table.to_csv(OUTPUT_DIR / file_name, encoding="utf-8-sig")


def rows_outside_limits(data, feature, limits):
    low, high = limits
    is_outside = data[feature].notna() & ~data[feature].between(low, high)
    return data[is_outside].copy()


def qrs_verdict(row):
    if row["past_next_R_ms"] > 0:
        return "klaida: QRS pabaiga uz kito duzio R"
    if row["into_T_wave_ms"] > 0:
        return "klaida: QRS pabaiga T bangoje"
    if row["QRS"] < QRS_LIMITS[0] and row["amplitude_ratio"] < 0.5:
        return "itartina: per maza QRS amplitude"
    if pd.isna(row["T_on"]):
        return "negalima patikrinti: nerasta T banga"
    return "priestaravimu nera: gali buti tikra reiksme"


def qt_verdict(row):
    if row["past_next_R_ms"] > 0:
        return "klaida: T pabaiga uz kito duzio R"
    if pd.isna(row["RR_post"]):
        return "negalima patikrinti: nera RR_post"
    return "priestaravimu nera: gali buti tikra reiksme"


def check_qrs(data):
    qrs = rows_outside_limits(data, "QRS", QRS_LIMITS)

    qrs["into_T_wave_ms"] = samples_to_ms(qrs["QRS_off"] - qrs["T_on"]).round(1)

    qrs_end_after_R_ms = samples_to_ms(qrs["QRS_off"] - qrs["R"])
    qrs["past_next_R_ms"] = (qrs_end_after_R_ms - qrs["RR_post"]).round(1)

    amplitude = data["QRS_max"] - data["QRS_min"]
    median_amplitude_by_class = amplitude.groupby(data["klase"]).median()
    qrs_amplitude = qrs["QRS_max"] - qrs["QRS_min"]
    class_median_amplitude = qrs["klase"].map(median_amplitude_by_class)
    qrs["amplitude_ratio"] = (qrs_amplitude / class_median_amplitude).round(2)

    qrs["verdict"] = qrs.apply(qrs_verdict, axis=1)

    return qrs


def check_qt(data):
    qt = rows_outside_limits(data, "QT", QT_LIMITS)

    t_end_after_R_ms = samples_to_ms(qt["T_off"] - qt["R"])
    qt["past_next_R_ms"] = (t_end_after_R_ms - qt["RR_post"]).round(1)

    qt["QT_to_RR_ratio"] = (qt["QT"] / qt["RR_post"]).round(2)

    qt["verdict"] = qt.apply(qt_verdict, axis=1)

    return qt


def report_feature(table, feature, limits, columns):
    low, high = limits
    file_prefix = f"14_{feature.lower()}"

    print(f"\n\n{feature} UŽ KONTROLINIŲ RIBŲ ({low}-{high} ms): {len(table)} atvejai")
    print(table[columns].sort_values(feature).to_string(index=False))
    save(table[columns].set_index(["irasas", "R"]), f"{file_prefix}_patikra.csv")

    print(f"\n{feature} išvados:")
    print(table["verdict"].value_counts().to_frame("atveju").to_string())

    verdicts_by_class = (
        pd.crosstab(table["klase"], table["verdict"])
        .reindex(CLASSES)
        .fillna(0)
        .astype(int)
    )
    print(f"\n{feature} išvados pagal klasę:")
    print(verdicts_by_class.to_string())
    save(verdicts_by_class, f"{file_prefix}_isvados.csv")


def summarize(table, feature, value_count):
    verdicts = table["verdict"]

    return {
        "feature": feature,
        "values": value_count,
        "outside_limits": len(table),
        "proven_errors": verdicts.str.startswith("klaida").sum(),
        "suspicious": verdicts.str.startswith("itartina").sum(),
        "uncheckable": verdicts.str.startswith("negalima").sum(),
        "possibly_real": verdicts.str.startswith("priestaravimu").sum(),
    }


if __name__ == "__main__":
    data = pd.read_csv(DATA_FILE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    qrs = check_qrs(data)
    report_feature(qrs, "QRS", QRS_LIMITS, QRS_COLUMNS)

    qt = check_qt(data)
    report_feature(qt, "QT", QT_LIMITS, QT_COLUMNS)

    qt_by_record = pd.crosstab(qt["irasas"], qt["verdict"])
    print("\nQT už ribų pagal įrašą ir išvadą:")
    print(qt_by_record.to_string())
    save(qt_by_record, "14_qt_pagal_irasa.csv")

    summary = pd.DataFrame([
        summarize(qrs, "QRS", data["QRS"].notna().sum()),
        summarize(qt, "QT", data["QT"].notna().sum()),
    ]).set_index("feature")

    summary["outside_limits_pct"] = (summary["outside_limits"] / summary["values"] * 100).round(2)
    summary["proven_errors_pct"] = (summary["proven_errors"] / summary["outside_limits"] * 100).round(1)

    print("\n\nSANTRAUKA:")
    print(summary.to_string())
    save(summary, "14_santrauka.csv")
