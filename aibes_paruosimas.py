
#Replaces invalid PR values with NaN and removes rows where any candidate feature is missing.
import pandas as pd

SAMPLE_FILE = "duomenys/tiriamoji.csv"
CLEANED_FILE = "duomenys/tiriamoji_sutvarkyta.csv"
COMPLETE_ROWS_FILE = "duomenys/tiriamoji_analizei.csv"
CLASS = "klase"

CANDIDATE_FEATURES = [
    "QRS",
    "RR_sant_vid",
    "RR_sant_post",
    "RR_pre",
    "QTc",
    "P_yra",
    "T_tipas",
]


def replace_invalid_pr(sample):
    cleaned = sample.copy()
    is_invalid_pr = cleaned["PR"].notna() & (cleaned["PR"] <= 0)

    print("PR reikšmių, pakeičiamų į NaN:", is_invalid_pr.sum())

    cleaned.loc[is_invalid_pr, "PR"] = pd.NA
    return cleaned


def print_candidate_missing_values(cleaned):
    candidates = cleaned[CANDIDATE_FEATURES]

    missing = pd.DataFrame({
        "count": candidates.isna().sum(),
        "percent": candidates.isna().mean() * 100,
    })

    print("\npozymiu trukstamumas:")
    print(missing.round(2))


def print_incomplete_rows(cleaned):
    has_missing_value = cleaned[CANDIDATE_FEATURES].isna().any(axis=1)

    print("\eilutes, kuriose trūksta bent vieno kandidatinio požymio:")
    print("Kiekis:", has_missing_value.sum())
    print("Procentai:", round(has_missing_value.mean() * 100, 2))

    print("\ntrukstamos eilutes pagal klase:")
    print(
        has_missing_value
        .groupby(cleaned[CLASS])
        .agg(["sum", "count", "mean"])
        .assign(percent=lambda table: table["mean"] * 100)
        [["sum", "count", "percent"]]
        .round(2)
    )

    print("\npilnos eilutes klasems:")
    print((~has_missing_value).groupby(cleaned[CLASS]).sum())


def keep_complete_rows(cleaned):
    complete_rows = cleaned.dropna(subset=CANDIDATE_FEATURES).copy()

    print("\npilnu eiluciu aibe:")
    print("Eilučių prieš:", len(cleaned))
    print("Eilučių po:", len(complete_rows))
    print("Pašalinta:", len(cleaned) - len(complete_rows))

    print("\nklasiu pasiskirstymas pilnuose duomenyse:")
    print(complete_rows[CLASS].value_counts().sort_index())

    print("\nklasiu proporcijos:")
    class_shares = complete_rows[CLASS].value_counts(normalize=True).sort_index()
    print((class_shares * 100).round(2))

    return complete_rows


def main():
    sample = pd.read_csv(SAMPLE_FILE)

    cleaned = replace_invalid_pr(sample)
    cleaned.to_csv(CLEANED_FILE, index=False)

    print_candidate_missing_values(cleaned)
    print_incomplete_rows(cleaned)

    complete_rows = keep_complete_rows(cleaned)
    complete_rows.to_csv(COMPLETE_ROWS_FILE, index=False)


if __name__ == "__main__":
    main()