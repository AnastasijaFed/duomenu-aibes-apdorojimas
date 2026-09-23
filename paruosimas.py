import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("duomenys/duziai.csv")

CLASSES = ["N", "L", "R", "V", "A"]

# Paliekame tik tiriamas klases ir pašaliname 114 įrašą,
# nes ECGPUWAVE jame analizavo kitą derivaciją nei kituose įrašuose.
df_5 = df[
    df["klase"].isin(CLASSES)
    & (df["irasas"].astype(str) != "114")
].copy()

# Parodome, iš kurių įrašų ateina kiekvienos klasės dūžiai
print("PRIEŠ ATRANKĄ:")
print(pd.crosstab(df_5["irasas"], df_5["klase"], margins=True))

# Atsitiktinai atrenkame po 1000 dūžių iš kiekvienos klasės
sample_df = (
    df_5
    .groupby("klase", group_keys=False)
    .sample(n=1000, random_state=42)
    .reset_index(drop=True)
)

# Aiškiai atskiriame du skirtingus RR intervalų santykius.
#
# Pradiniame duomenų faile stulpelis RR_sant jau reiškia:
# RR_pre / RR_vid.
# Jį paliekame nepakeistą, kad nesugadintume ankstesnio duomenų formato,
# bet analizei sukuriame aiškiai pavadintą kopiją RR_sant_vid.
sample_df["RR_sant_vid"] = sample_df["RR_sant"]

# Antras santykis lygina RR intervalą prieš dūžį su intervalu po dūžio.
sample_df["RR_sant_post"] = (
    sample_df["RR_pre"] / sample_df["RR_post"]
)

# Išsaugome tiriamąją aibę jau su abiem RR santykiais.
sample_df.to_csv("duomenys/tiriamoji.csv", index=False)

print("\nPO ATRANKOS:")
print(sample_df["klase"].value_counts())

print("\nDŪŽIAI PAGAL ĮRAŠĄ IR KLASĘ:")
print(pd.crosstab(
    sample_df["irasas"],
    sample_df["klase"],
    margins=True
))

print("\nTRŪKSTAMOS REIKŠMĖS:")

# Kol kas tik visi analizei aktualūs požymiai
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
    "T_tipas"
]

# Bendras trūkstamų reikšmių kiekis ir procentas
missing = pd.DataFrame({
    "kiekis": sample_df[FEATURES].isna().sum(),
    "procentai": sample_df[FEATURES].isna().mean() * 100
})

print("\nBendrai:")
print(missing.round(2))

# Trūkstamų reikšmių procentas pagal klasę
missing_by_class = (
    sample_df
    .groupby("klase")[FEATURES]
    .apply(lambda x: x.isna().mean() * 100)
)

print("\nPagal klases (%):")
print(missing_by_class.round(2))

print("\nECGPUWAVE DŪŽIŲ SUDERINIMAS:")

found_by_class = (
    sample_df.groupby("klase")["rastas"]
    .agg(["sum", "count", "mean"])
)

found_by_class["procentai"] = (
    found_by_class["mean"] * 100
)

print(
    found_by_class[
        ["sum", "count", "procentai"]
    ].round(2)
)


print("\nP BANGA PAGAL KLASĘ (%):")
print("-1 = dūžis nesuderintas, 0 = P banga neaptikta, 1 = P banga aptikta")

p_wave = (
    pd.crosstab(
        sample_df["klase"],
        sample_df["P_yra"].fillna(-1),
        normalize="index"
    ) * 100
)

print(p_wave.round(1))


print("\nTRŪKSTAMOS REIKŠMĖS PAGAL 'rastas':")

for col in ["PR", "QRS", "QT", "P_amp", "T_amp", "T_tipas"]:
    table = pd.crosstab(
        sample_df["rastas"],
        sample_df[col].isna()
    )

    print(f"\n{col}:")
    print(table)


print("\nN DŪŽIAI BE P BANGOS PAGAL RITMĄ:")

# Naudojame visus N klasės dūžius be 114 įrašo,
# nes 1000 dūžių imtyje reti ritmai būtų menkai atstovaujami.
n_beats = df_5[
    (df_5["klase"] == "N")
    & df_5["P_yra"].notna()
].copy()

p_missing = (
    (n_beats["P_yra"] == 0)
    .groupby(n_beats["ritmas"], dropna=False)
    .agg(["mean", "size"])
)

p_missing["mean"] = (
    p_missing["mean"] * 100
).round(1)

p_missing = p_missing.rename(
    columns={
        "mean": "P_nerasta_%",
        "size": "duziu"
    }
)

print(
    p_missing.sort_values(
        "duziu",
        ascending=False
    )
)

print("\nP NERASTA N DŪŽIUOSE SINUSINIAME RITME PAGAL ĮRAŠĄ:")

sinus = n_beats[n_beats["ritmas"] == "(N"].copy()

p_by_record = (
    (sinus["P_yra"] == 0)
    .groupby(sinus["irasas"])
    .agg(["mean", "size"])
)

p_by_record["mean"] = (
    p_by_record["mean"] * 100
).round(1)

p_by_record = p_by_record.rename(
    columns={
        "mean": "P_nerasta_%",
        "size": "duziu"
    }
)

print(p_by_record.sort_values("P_nerasta_%"))


print("\nDUBLIKATŲ PATIKRA:")

duplicates = sample_df.duplicated(
    subset=["irasas", "R"]
).sum()

print("Dublikatų (irasas + R):", duplicates)


print("\nNELOGIŠKOS REIKŠMĖS (<= 0):")

for col in [
    "RR_pre",
    "RR_post",
    "RR_vid",
    "RR_sant_vid",
    "RR_sant_post",
    "PR",
    "QRS",
    "QT",
    "QTc"
]:
    count = (sample_df[col] <= 0).sum()
    print(col, count)


print("\nUŽ KONTROLINIŲ RIBŲ PAGAL KLASĘ:")

LIMITS = {
    "PR": (50, 400),
    "QRS": (40, 250),
    "QT": (200, 700)
}

for col, (low, high) in LIMITS.items():

    out_of_range = (
        sample_df[col].notna()
        & ~sample_df[col].between(low, high)
    )

    print(f"\n{col} ({low}–{high} ms):")

    print(
        out_of_range
        .groupby(sample_df["klase"])
        .agg(["sum", "count"])
    )

print("\nUŽ RIBŲ: ŽEMIAU / AUKŠČIAU:")

for col, (low, high) in LIMITS.items():
    x = sample_df[col]

    print(
        f"{col}: "
        f"< {low}: {(x < low).sum()}, "
        f"> {high}: {(x > high).sum()}, "
        f"min = {x.min():.1f}, "
        f"max = {x.max():.1f}"
    )


print("\nIŠSKIRTYS PAGAL IQR TAISYKLĘ KLASĖS VIDUJE (%):")

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
    "QRS_min"
]


def iqr_outlier_pct(x):
    # Trūkstamų reikšmių į išskirčių skaičiavimą neįtraukiame
    x = x.dropna()

    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = (
        (x < lower_bound)
        | (x > upper_bound)
    )

    return outliers.mean() * 100


outliers_by_class = (
    sample_df
    .groupby("klase")[NUMERIC_FEATURES]
    .agg(iqr_outlier_pct)
    .T
)

print(outliers_by_class.round(1))


print("\nABIEJŲ RR SANTYKIŲ PALYGINIMAS PAGAL KLASĘ:")

for col in ["RR_sant_vid", "RR_sant_post"]:
    print(f"\n{col}:")
    print(
        sample_df
        .groupby("klase")[col]
        .agg(
            kiekis="count",
            vidurkis="mean",
            mediana="median",
            std="std",
            min="min",
            max="max"
        )
        .round(3)
    )


print("\nĮTARTINOS QRS REIKŠMĖS:")

qrs_suspicious = sample_df[
    sample_df["QRS"].notna()
    & ~sample_df["QRS"].between(40, 250)
][["irasas", "R", "klase", "QRS", "QRS_on", "QRS_off"]]

print(
    qrs_suspicious
    .sort_values("QRS")
    .to_string(index=False)
)


print("\nĮTARTINOS QT REIKŠMĖS:")

qt_suspicious = sample_df[
    sample_df["QT"].notna()
    & ~sample_df["QT"].between(200, 700)
][["irasas", "R", "klase", "QT", "QRS_on", "T_off"]]

print(
    qt_suspicious
    .sort_values("QT")
    .to_string(index=False)
)


print("\nQRS UŽ RIBŲ PAGAL ĮRAŠĄ:")

print(
    qrs_suspicious
    .groupby(["irasas", "klase"])
    .size()
    .sort_values(ascending=False)
)


print("\nQT UŽ RIBŲ PAGAL ĮRAŠĄ:")

print(
    qt_suspicious
    .groupby(["irasas", "klase"])
    .size()
    .sort_values(ascending=False)
)

print("\nLOGINĖ ANOTACIJŲ PATIKRA:")

# Kito dūžio R vietą randame PILNOJE aibėje,
# nes tiriamoji aibė yra atsitiktinė 5000 dūžių imtis.
df_with_next_r = (
    df
    .sort_values(["irasas", "R"])
    .copy()
)

df_with_next_r["R_kitas"] = (
    df_with_next_r
    .groupby("irasas")["R"]
    .shift(-1)
)

check = sample_df.merge(
    df_with_next_r[["irasas", "R", "R_kitas"]],
    on=["irasas", "R"],
    how="left"
)


# 1. R taškas turi būti QRS komplekso viduje
qrs_excludes_r = (
    check["QRS_on"].notna()
    & check["QRS_off"].notna()
    & (
        (check["R"] < check["QRS_on"])
        | (check["R"] > check["QRS_off"])
    )
)

print(
    "R nepatenka tarp QRS_on ir QRS_off:",
    qrs_excludes_r.sum()
)


# 2. Dabartinio QRS pabaiga neturėtų nueiti už kito R
qrs_past_next_r = (
    check["QRS_off"].notna()
    & check["R_kitas"].notna()
    & (check["QRS_off"] >= check["R_kitas"])
)

print(
    "QRS_off pasiekia / viršija kito dūžio R:",
    qrs_past_next_r.sum()
)


# 3. T pabaiga turi būti po QRS pabaigos
t_before_qrs_end = (
    check["T_off"].notna()
    & check["QRS_off"].notna()
    & (check["T_off"] <= check["QRS_off"])
)

print(
    "T_off yra prieš QRS_off arba sutampa:",
    t_before_qrs_end.sum()
)


# 4. Patikriname, kiek T bangų tęsiasi iki kito R ar už jo
t_past_next_r = (
    check["T_off"].notna()
    & check["R_kitas"].notna()
    & (check["T_off"] >= check["R_kitas"])
)

print(
    "T_off pasiekia / viršija kito dūžio R:",
    t_past_next_r.sum()
)


# 5. PR loginė patikra
p_after_qrs = (
    check["P_on"].notna()
    & check["QRS_on"].notna()
    & (check["P_on"] >= check["QRS_on"])
)

print(
    "P_on yra ties QRS_on arba po jo:",
    p_after_qrs.sum()
)


print("\nĮTARTINI ATVEJAI PAGAL KLASĘ:")

print(pd.DataFrame({
    "QRS_neapima_R": qrs_excludes_r.groupby(check["klase"]).sum(),
    "QRS_uz_kito_R": qrs_past_next_r.groupby(check["klase"]).sum(),
    "T_pries_QRS_off": t_before_qrs_end.groupby(check["klase"]).sum(),
    "T_uz_kito_R": t_past_next_r.groupby(check["klase"]).sum(),
    "P_on_po_QRS_on": p_after_qrs.groupby(check["klase"]).sum(),
}))

print("\nPATIKSLINTA QRS LOGINĖ PATIKRA:")

# QRS_on ir QRS_off yra ECGPUWAVE anotacijos,
# todėl tikriname ECGPUWAVE aptiktą R_ecg, o ne ekspertinį R.
qrs_excludes_r_ecg = (
    check["R_ecg"].notna()
    & check["QRS_on"].notna()
    & check["QRS_off"].notna()
    & (
        (check["R_ecg"] < check["QRS_on"])
        | (check["R_ecg"] > check["QRS_off"])
    )
)

print(
    "R_ecg nepatenka tarp QRS_on ir QRS_off:",
    qrs_excludes_r_ecg.sum()
)


# Kiek skiriasi ekspertinis R nuo ECGPUWAVE aptikto R
r_distance = (
    check.loc[
        check["R_ecg"].notna(),
        "R"
    ]
    - check.loc[
        check["R_ecg"].notna(),
        "R_ecg"
    ]
).abs()

print("\n|R - R_ecg| atstumas mėginiais:")
print(r_distance.describe().round(2))


print("\n|R - R_ecg| pagal klasę:")

check["R_atstumas"] = (
    check["R"] - check["R_ecg"]
).abs()

print(
    check
    .groupby("klase")["R_atstumas"]
    .agg(["count", "mean", "median", "max"])
    .round(2)
)


print("\nANKSTESNI 72 ATVEJAI - JŲ R ATSTUMAS:")

print(
    check.loc[
        qrs_excludes_r,
        ["irasas", "klase", "R", "R_ecg",
         "R_atstumas", "QRS_on", "QRS_off"]
    ]
    .sort_values("R_atstumas", ascending=False)
    .to_string(index=False)
)

print("\nPIRMINIS DUOMENŲ TVARKYMAS:")

cleaned = sample_df.copy()

# Aiškiai nelogiškos PR reikšmės:
# P pradžia negali būti ties QRS pradžia arba po jos.
invalid_pr = (
    cleaned["PR"].notna()
    & (cleaned["PR"] <= 0)
)

print("PR reikšmių, pakeičiamų į NaN:", invalid_pr.sum())

cleaned.loc[invalid_pr, "PR"] = pd.NA

# Išsaugome atskirai – pradinės imties neperrašome.
cleaned.to_csv(
    "duomenys/tiriamoji_sutvarkyta.csv",
    index=False
)

print("\nKANDIDATINIŲ POŽYMIŲ TRŪKSTAMUMAS PO TVARKYMO:")

# Šie požymiai kol kas yra kandidatai išsamesnei analizei.
# Galutinius 3–4 požymius pasirinksime tik palyginę jų kokybę
# ir pasiskirstymą tarp klasių.
CANDIDATE_FEATURES = [
    "QRS",
    "RR_sant_vid",
    "RR_sant_post",
    "RR_pre",
    "QTc",
    "P_yra",
    "T_tipas"
]

missing_candidates = pd.DataFrame({
    "kiekis": cleaned[CANDIDATE_FEATURES].isna().sum(),
    "procentai": cleaned[CANDIDATE_FEATURES].isna().mean() * 100
})

print(missing_candidates.round(2))


print("\nEILUTĖS, KURIOSE TRŪKSTA BENT VIENO KANDIDATINIO POŽYMIO:")

missing_any = (
    cleaned[CANDIDATE_FEATURES]
    .isna()
    .any(axis=1)
)

print("Kiekis:", missing_any.sum())
print(
    "Procentai:",
    round(missing_any.mean() * 100, 2)
)


print("\nTRŪKSTAMOS EILUTĖS PAGAL KLASĘ:")

print(
    missing_any
    .groupby(cleaned["klase"])
    .agg(["sum", "count", "mean"])
    .assign(procentai=lambda x: x["mean"] * 100)
    [["sum", "count", "procentai"]]
    .round(2)
)


print("\nPILNOS EILUTĖS PAGAL KLASĘ:")

complete_row = ~missing_any

print(
    complete_row
    .groupby(cleaned["klase"])
    .sum()
)

print("\nPILNŲ EILUČIŲ AIBĖ KANDIDATINIŲ POŽYMIŲ ANALIZEI:")

analysis_df = (
    cleaned
    .dropna(subset=CANDIDATE_FEATURES)
    .copy()
)

print("Eilučių prieš:", len(cleaned))
print("Eilučių po:", len(analysis_df))
print(
    "Pašalinta:",
    len(cleaned) - len(analysis_df)
)

print("\nKLASIŲ PASISKIRSTYMAS PO TRŪKSTAMŲ EILUČIŲ PAŠALINIMO:")
print(analysis_df["klase"].value_counts().sort_index())

print("\nKLASIŲ PROPORCIJOS (%):")
print(
    (
        analysis_df["klase"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    ).round(2)
)

analysis_df.to_csv(
    "duomenys/tiriamoji_analizei.csv",
    index=False
)

print("\nAPRAŠOMOJI STATISTIKA PAGAL KLASĘ:")

CANDIDATE_NUMERIC = [
    "QRS",
    "RR_sant_vid",
    "RR_sant_post",
    "RR_pre",
    "QTc"
]


def descriptive_stats(x):
    return pd.Series({
        "n": x.count(),
        "vidurkis": x.mean(),
        "std": x.std(),
        "min": x.min(),
        "Q1": x.quantile(0.25),
        "mediana": x.median(),
        "Q3": x.quantile(0.75),
        "max": x.max()
    })


for col in CANDIDATE_NUMERIC:
    print(f"\n{col}:")

    table = (
        analysis_df
        .groupby("klase")[col]
        .apply(descriptive_stats)
        .unstack()
    )

    print(table.round(2))


print("\nP_YRA PASISKIRSTYMAS PAGAL KLASĘ (%):")

p_present_pct = (
    pd.crosstab(
        analysis_df["klase"],
        analysis_df["P_yra"],
        normalize="index"
    ) * 100
)

print(p_present_pct.round(1))


print("\nT_TIPAS PASISKIRSTYMAS PAGAL KLASĘ (%):")

t_type_pct = (
    pd.crosstab(
        analysis_df["klase"],
        analysis_df["T_tipas"],
        normalize="index"
    ) * 100
)

print(t_type_pct.round(1))

print("\nKURIAMOS PAGRINDINIŲ POŽYMIŲ VIZUALIZACIJOS:")

CLASS_ORDER = ["N", "L", "R", "V", "A"]

for col in ["QRS", "RR_sant_vid", "RR_sant_post", "RR_pre", "QTc"]:
    data = [
        analysis_df.loc[analysis_df["klase"] == cls, col]
        for cls in CLASS_ORDER
    ]

    plt.figure(figsize=(8, 5))

    plt.boxplot(
        data,
        tick_labels=CLASS_ORDER,
        showfliers=True
    )

    plt.title(f"{col} pasiskirstymas pagal klasę")
    plt.xlabel("Dūžio klasė")
    plt.ylabel(col)
    plt.tight_layout()

    plt.show()


# P_yra pasiskirstymas
p_present_plot = (
    pd.crosstab(
        analysis_df["klase"],
        analysis_df["P_yra"],
        normalize="index"
    )
    .reindex(CLASS_ORDER)
    * 100
)

p_present_plot.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 5),
    rot=0
)

plt.title("P bangos aptikimas pagal klasę")
plt.xlabel("Dūžio klasė")
plt.ylabel("Procentai")
plt.legend(
    ["P neaptikta", "P aptikta"],
    title="P_yra"
)
plt.tight_layout()
plt.show()


# T_tipas pasiskirstymas
# Skaitinius ECGPUWAVE T bangos tipo kodus pakeičiame aiškiais pavadinimais.
T_TYPES = {
    0.0: "normali",
    1.0: "apversta",
    2.0: "tik teigiama",
    3.0: "tik neigiama",
    4.0: "dvifazė −/+",
    5.0: "dvifazė +/−"
}

t_type_plot = (
    pd.crosstab(
        analysis_df["klase"],
        analysis_df["T_tipas"],
        normalize="index"
    )
    .reindex(CLASS_ORDER)
    .rename(columns=T_TYPES)
    * 100
)

t_type_plot.plot(
    kind="bar",
    stacked=True,
    figsize=(9, 5),
    rot=0
)

plt.title("T bangos tipo pasiskirstymas pagal klasę")
plt.xlabel("Dūžio klasė")
plt.ylabel("Procentai")
plt.legend(title="T bangos tipas")
plt.tight_layout()
plt.show()

print("\nASIMETRIJOS KOEFICIENTAS PAGAL KLASĘ:")
print(
    analysis_df
    .groupby("klase")[CANDIDATE_NUMERIC]
    .skew()
    .round(2)
)

print("\nMASTELIAI (min–max):")
print(
    analysis_df[CANDIDATE_NUMERIC]
    .agg(["min", "max"])
    .round(2)
)

print("\nT_TIPAS KIEKIAI (retos kategorijos):")
print(
    pd.crosstab(
        analysis_df["klase"],
        analysis_df["T_tipas"]
    )
)


# Patikriname hipotezę, ar 232 įrašas lemia platesnį
# A klasės QRS pasiskirstymą.
is_232 = analysis_df["irasas"].astype(str) == "232"
is_a = analysis_df["klase"] == "A"

print("\nA KLASĖ: 232 ĮRAŠAS (True) VS KITI (False), MEDIANOS:")
print(
    analysis_df[is_a]
    .groupby(is_232[is_a])[["QRS", "RR_post"]]
    .median()
    .round(1)
)


# Patikriname, iš kokių įrašų ateina labai ilgi
# R klasės RR_pre intervalai.
r_class = analysis_df[analysis_df["klase"] == "R"]

print("\nR KLASĖS RR_pre > 2000 ms PAGAL ĮRAŠĄ:")
print(
    r_class.loc[
        r_class["RR_pre"] > 2000,
        "irasas"
    ].value_counts()
)


# Patikriname, ar N klasės RR_sant_vid išskirtys
# dažniau susijusios su AFIB ritmu.
n_class = analysis_df[analysis_df["klase"] == "N"]

q1 = n_class["RR_sant_vid"].quantile(0.25)
q3 = n_class["RR_sant_vid"].quantile(0.75)
iqr = q3 - q1

is_outlier = ~n_class["RR_sant_vid"].between(
    q1 - 1.5 * iqr,
    q3 + 1.5 * iqr
)

is_afib = n_class["ritmas"] == "(AFIB"

print(
    f"\nN RR_sant_vid: AFIB dalis tarp išskirčių "
    f"{is_afib[is_outlier].mean() * 100:.1f} %, "
    f"tarp kitų {is_afib[~is_outlier].mean() * 100:.1f} %"
)