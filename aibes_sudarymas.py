import pandas as pd

all_beats_file = "duomenys/duziai.csv"
sample_file = "duomenys/tiriamoji.csv"

CLASS = "klase"
RECORD = "irasas"

CLASSES = ["N", "L", "R", "V", "A"]
BEATS_PER_CLASS = 1000
RANDOM_STATE = 42

EXCLUDED_RECORD = "114"


def select_study_beats(all_beats):
    is_studied_class = all_beats[CLASS].isin(CLASSES)
    is_excluded_record = all_beats[RECORD].astype(str) == EXCLUDED_RECORD
    return all_beats[is_studied_class & ~is_excluded_record].copy()


def draw_stratified_sample(beats):
    return (
        beats
        .groupby(CLASS, group_keys=False)
        .sample(n=BEATS_PER_CLASS, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )


def add_rr_ratios(sample):
    sample["RR_sant_vid"] = sample["RR_sant"]
    sample["RR_sant_post"] = sample["RR_pre"] / sample["RR_post"]
    return sample


def main():
    all_beats = pd.read_csv(all_beats_file)
    study_beats = select_study_beats(all_beats)

    sample = draw_stratified_sample(study_beats)
    sample = add_rr_ratios(sample)
    sample.to_csv(sample_file, index=False)

if __name__ == "__main__":
    main()