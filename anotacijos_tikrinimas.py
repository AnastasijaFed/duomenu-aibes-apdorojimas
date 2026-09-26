import pandas as pd

all_beats_file = "duomenys/duziai.csv"
sample_file = "duomenys/tiriamoji.csv"

class_column = "klase"
record_column = "irasas"

#normal beat order: p_on < qrs_on < r < qrs_off < t_off < next_r
def add_next_r_peak(all_beats, sample):
    beats_in_order = all_beats.sort_values([record_column, "R"]).copy()
    beats_in_order["next_R"] = beats_in_order.groupby(record_column)["R"].shift(-1)

    return sample.merge(
        beats_in_order[[record_column, "R", "next_R"]],
        on=[record_column, "R"],
        how="left",
    )


def is_known(beats, *columns):#checking if value exists
    known = beats[columns[0]].notna()
    for column in columns[1:]:
        known &= beats[column].notna()
    return known


def find_annotation_errors(beats): #checking rules
    return pd.DataFrame({
        # R peak must lie inside the QRS complex
        "R_outside_QRS": (
            is_known(beats, "QRS_on", "QRS_off")
            & ((beats["R"] < beats["QRS_on"]) | (beats["R"] > beats["QRS_off"]))
        ),
        # QRS complex must end before the next beat's R peak
        "QRS_past_next_R": (
            is_known(beats, "QRS_off", "next_R")
            & (beats["QRS_off"] >= beats["next_R"])
        ),
        # T wave must end after the QRS complex ends
        "T_before_QRS_end": (
            is_known(beats, "T_off", "QRS_off")
            & (beats["T_off"] <= beats["QRS_off"])
        ),
        # T wave should end before the next beat's R peak
        "T_past_next_R": (
            is_known(beats, "T_off", "next_R")
            & (beats["T_off"] >= beats["next_R"])
        ),
        # P wave must start before the QRS complex
        "P_after_QRS_start": (
            is_known(beats, "P_on", "QRS_on")
            & (beats["P_on"] >= beats["QRS_on"])
        ),
    })


def print_annotation_errors(beats, errors):
    print("Loginė anotacijų patikra:")
    print("R nepatenka tarp QRS_on ir QRS_off:", errors["R_outside_QRS"].sum())
    print("QRS_off pasiekia / viršija kito dūžio R:", errors["QRS_past_next_R"].sum())
    print("T_off yra prieš QRS_off arba sutampa:", errors["T_before_QRS_end"].sum())
    print("T_off pasiekia / viršija kito dūžio R:", errors["T_past_next_R"].sum())
    print("P_on yra ties QRS_on arba po jo:", errors["P_after_QRS_start"].sum())

    print("\nĮtartini atvejai pagal klasę:")
    print(errors.groupby(beats[class_column]).sum())


def print_refined_qrs_check(beats):
    ecg_r_outside_qrs = (
        is_known(beats, "R_ecg", "QRS_on", "QRS_off")
        & ((beats["R_ecg"] < beats["QRS_on"]) | (beats["R_ecg"] > beats["QRS_off"]))
    )

    print("\nPatikslinta QRS loginė patikra:")
    print("R_ecg nepatenka tarp QRS_on ir QRS_off:", ecg_r_outside_qrs.sum())


def print_r_peak_distances(beats, errors):
    beats["R_distance"] = (beats["R"] - beats["R_ecg"]).abs()

    print("\n|R - R_ecg| atstumas mėginiais:")
    print(beats["R_distance"].dropna().describe().round(2))

    print("\n|R - R_ecg| pagal klasę:")
    print(
        beats
        .groupby(class_column)["R_distance"]
        .agg(["count", "mean", "median", "max"])
        .round(2)
    )

    #beats where the expert's R fell outside the QRS complex: large R_distance shows that the error comes from the two R peaks
    #being far apart, not from a wrong QRS annotation
    print("\nAnkstesni 72 atvejai - jų R atstumas:")
    print(
        beats.loc[
            errors["R_outside_QRS"],
            [record_column, class_column, "R", "R_ecg", "R_distance", "QRS_on", "QRS_off"],
        ]
        .sort_values("R_distance", ascending=False)
        .to_string(index=False)
    )


def main():
    all_beats = pd.read_csv(all_beats_file)
    sample = pd.read_csv(sample_file)

    beats = add_next_r_peak(all_beats, sample)
    errors = find_annotation_errors(beats)

    print_annotation_errors(beats, errors)
    print_refined_qrs_check(beats)
    print_r_peak_distances(beats, errors)


if __name__ == "__main__":
    main()