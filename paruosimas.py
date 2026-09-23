import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("duomenys/duziai.csv")

KLASES = ["N", "L", "R", "V", "A"]

# Paliekame tik tiriamas klases ir pašaliname 114 įrašą,
# nes ECGPUWAVE jame analizavo kitą derivaciją nei kituose įrašuose.
df_5 = df[
    df["klase"].isin(KLASES)
    & (df["irasas"].astype(str) != "114")
].copy()

# Parodome, iš kurių įrašų ateina kiekvienos klasės dūžiai
print("PRIEŠ ATRANKĄ:")
print(pd.crosstab(df_5["irasas"], df_5["klase"], margins=True))

# Atsitiktinai atrenkame po 1000 dūžių iš kiekvienos klasės
tiriamoji = (
    df_5
    .groupby("klase", group_keys=False)
    .sample(n=1000, random_state=42)
    .reset_index(drop=True)
)

# Išsaugome tiriamąją aibę
tiriamoji.to_csv("duomenys/tiriamoji.csv", index=False)

print("\nPO ATRANKOS:")
print(tiriamoji["klase"].value_counts())

print("\nDŪŽIAI PAGAL ĮRAŠĄ IR KLASĘ:")
print(pd.crosstab(
    tiriamoji["irasas"],
    tiriamoji["klase"],
    margins=True
))

print("\nTRŪKSTAMOS REIKŠMĖS:")

# Kol kas tik visi analizei aktualūs požymiai
POZYMIAI = [
    "RR_pre",
    "RR_post",
    "RR_vid",
    "RR_sant",
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
trukstamos = pd.DataFrame({
    "kiekis": tiriamoji[POZYMIAI].isna().sum(),
    "procentai": tiriamoji[POZYMIAI].isna().mean() * 100
})

print("\nBendrai:")
print(trukstamos.round(2))

# Trūkstamų reikšmių procentas pagal klasę
trukstamos_pagal_klase = (
    tiriamoji
    .groupby("klase")[POZYMIAI]
    .apply(lambda x: x.isna().mean() * 100)
)

print("\nPagal klases (%):")
print(trukstamos_pagal_klase.round(2))

print("\nECGPUWAVE DŪŽIŲ SUDERINIMAS:")

rasta_pagal_klase = (
    tiriamoji.groupby("klase")["rastas"]
    .agg(["sum", "count", "mean"])
)

rasta_pagal_klase["procentai"] = (
    rasta_pagal_klase["mean"] * 100
)

print(
    rasta_pagal_klase[
        ["sum", "count", "procentai"]
    ].round(2)
)


print("\nP BANGA PAGAL KLASĘ (%):")
print("-1 = dūžis nesuderintas, 0 = P banga neaptikta, 1 = P banga aptikta")

p_banga = (
    pd.crosstab(
        tiriamoji["klase"],
        tiriamoji["P_yra"].fillna(-1),
        normalize="index"
    ) * 100
)

print(p_banga.round(1))


print("\nTRŪKSTAMOS REIKŠMĖS PAGAL 'rastas':")

for col in ["PR", "QRS", "QT", "P_amp", "T_amp", "T_tipas"]:
    lentele = pd.crosstab(
        tiriamoji["rastas"],
        tiriamoji[col].isna()
    )

    print(f"\n{col}:")
    print(lentele)


print("\nN DŪŽIAI BE P BANGOS PAGAL RITMĄ:")

# Naudojame visus N klasės dūžius be 114 įrašo,
# nes 1000 dūžių imtyje reti ritmai būtų menkai atstovaujami.
n = df_5[
    (df_5["klase"] == "N")
    & df_5["P_yra"].notna()
].copy()

p_truksta = (
    (n["P_yra"] == 0)
    .groupby(n["ritmas"], dropna=False)
    .agg(["mean", "size"])
)

p_truksta["mean"] = (
    p_truksta["mean"] * 100
).round(1)

p_truksta = p_truksta.rename(
    columns={
        "mean": "P_nerasta_%",
        "size": "duziu"
    }
)

print(
    p_truksta.sort_values(
        "duziu",
        ascending=False
    )
)

print("\nP NERASTA N DŪŽIUOSE SINUSINIAME RITME PAGAL ĮRAŠĄ:")

sin = n[n["ritmas"] == "(N"].copy()

p_pagal_irasa = (
    (sin["P_yra"] == 0)
    .groupby(sin["irasas"])
    .agg(["mean", "size"])
)

p_pagal_irasa["mean"] = (
    p_pagal_irasa["mean"] * 100
).round(1)

p_pagal_irasa = p_pagal_irasa.rename(
    columns={
        "mean": "P_nerasta_%",
        "size": "duziu"
    }
)

print(p_pagal_irasa.sort_values("P_nerasta_%"))


print("\nDUBLIKATŲ PATIKRA:")

dublikatai = tiriamoji.duplicated(
    subset=["irasas", "R"]
).sum()

print("Dublikatų (irasas + R):", dublikatai)


print("\nNELOGIŠKOS REIKŠMĖS (<= 0):")

for col in [
    "RR_pre",
    "RR_post",
    "RR_vid",
    "RR_sant",
    "PR",
    "QRS",
    "QT",
    "QTc"
]:
    kiek = (tiriamoji[col] <= 0).sum()
    print(col, kiek)


print("\nUŽ KONTROLINIŲ RIBŲ PAGAL KLASĘ:")

RIBOS = {
    "PR": (50, 400),
    "QRS": (40, 250),
    "QT": (200, 700)
}

for col, (nuo, iki) in RIBOS.items():

    uz_ribu = (
        tiriamoji[col].notna()
        & ~tiriamoji[col].between(nuo, iki)
    )

    print(f"\n{col} ({nuo}–{iki} ms):")

    print(
        uz_ribu
        .groupby(tiriamoji["klase"])
        .agg(["sum", "count"])
    )

print("\nUŽ RIBŲ: ŽEMIAU / AUKŠČIAU:")

for col, (nuo, iki) in RIBOS.items():
    x = tiriamoji[col]

    print(
        f"{col}: "
        f"< {nuo}: {(x < nuo).sum()}, "
        f"> {iki}: {(x > iki).sum()}, "
        f"min = {x.min():.1f}, "
        f"max = {x.max():.1f}"
    )


print("\nIŠSKIRTYS PAGAL IQR TAISYKLĘ KLASĖS VIDUJE (%):")

SKAITINIAI = [
    "RR_pre",
    "RR_post",
    "RR_vid",
    "RR_sant",
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


def iqr_isskirtys(x):
    # Trūkstamų reikšmių į išskirčių skaičiavimą neįtraukiame
    x = x.dropna()

    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1

    apatine_riba = q1 - 1.5 * iqr
    virsutine_riba = q3 + 1.5 * iqr

    isksirtys = (
        (x < apatine_riba)
        | (x > virsutine_riba)
    )

    return isksirtys.mean() * 100


isksirtys = (
    tiriamoji
    .groupby("klase")[SKAITINIAI]
    .agg(iqr_isskirtys)
    .T
)

print(isksirtys.round(1))

print("\nĮTARTINOS QRS REIKŠMĖS:")

qrs_itartinos = tiriamoji[
    tiriamoji["QRS"].notna()
    & ~tiriamoji["QRS"].between(40, 250)
][["irasas", "R", "klase", "QRS", "QRS_on", "QRS_off"]]

print(
    qrs_itartinos
    .sort_values("QRS")
    .to_string(index=False)
)


print("\nĮTARTINOS QT REIKŠMĖS:")

qt_itartinos = tiriamoji[
    tiriamoji["QT"].notna()
    & ~tiriamoji["QT"].between(200, 700)
][["irasas", "R", "klase", "QT", "QRS_on", "T_off"]]

print(
    qt_itartinos
    .sort_values("QT")
    .to_string(index=False)
)


print("\nQRS UŽ RIBŲ PAGAL ĮRAŠĄ:")

print(
    qrs_itartinos
    .groupby(["irasas", "klase"])
    .size()
    .sort_values(ascending=False)
)


print("\nQT UŽ RIBŲ PAGAL ĮRAŠĄ:")

print(
    qt_itartinos
    .groupby(["irasas", "klase"])
    .size()
    .sort_values(ascending=False)
)

print("\nLOGINĖ ANOTACIJŲ PATIKRA:")

# Kito dūžio R vietą randame PILNOJE aibėje,
# nes tiriamoji aibė yra atsitiktinė 5000 dūžių imtis.
df_su_kitu_r = (
    df
    .sort_values(["irasas", "R"])
    .copy()
)

df_su_kitu_r["R_kitas"] = (
    df_su_kitu_r
    .groupby("irasas")["R"]
    .shift(-1)
)

tikrinimas = tiriamoji.merge(
    df_su_kitu_r[["irasas", "R", "R_kitas"]],
    on=["irasas", "R"],
    how="left"
)


# 1. R taškas turi būti QRS komplekso viduje
qrs_neapima_r = (
    tikrinimas["QRS_on"].notna()
    & tikrinimas["QRS_off"].notna()
    & (
        (tikrinimas["R"] < tikrinimas["QRS_on"])
        | (tikrinimas["R"] > tikrinimas["QRS_off"])
    )
)

print(
    "R nepatenka tarp QRS_on ir QRS_off:",
    qrs_neapima_r.sum()
)


# 2. Dabartinio QRS pabaiga neturėtų nueiti už kito R
qrs_uz_kito_r = (
    tikrinimas["QRS_off"].notna()
    & tikrinimas["R_kitas"].notna()
    & (tikrinimas["QRS_off"] >= tikrinimas["R_kitas"])
)

print(
    "QRS_off pasiekia / viršija kito dūžio R:",
    qrs_uz_kito_r.sum()
)


# 3. T pabaiga turi būti po QRS pabaigos
t_pries_qrs_pabaiga = (
    tikrinimas["T_off"].notna()
    & tikrinimas["QRS_off"].notna()
    & (tikrinimas["T_off"] <= tikrinimas["QRS_off"])
)

print(
    "T_off yra prieš QRS_off arba sutampa:",
    t_pries_qrs_pabaiga.sum()
)


# 4. Patikriname, kiek T bangų tęsiasi iki kito R ar už jo
t_uz_kito_r = (
    tikrinimas["T_off"].notna()
    & tikrinimas["R_kitas"].notna()
    & (tikrinimas["T_off"] >= tikrinimas["R_kitas"])
)

print(
    "T_off pasiekia / viršija kito dūžio R:",
    t_uz_kito_r.sum()
)


# 5. PR loginė patikra
p_po_qrs = (
    tikrinimas["P_on"].notna()
    & tikrinimas["QRS_on"].notna()
    & (tikrinimas["P_on"] >= tikrinimas["QRS_on"])
)

print(
    "P_on yra ties QRS_on arba po jo:",
    p_po_qrs.sum()
)


print("\nĮTARTINI ATVEJAI PAGAL KLASĘ:")

print(pd.DataFrame({
    "QRS_neapima_R": qrs_neapima_r.groupby(tikrinimas["klase"]).sum(),
    "QRS_uz_kito_R": qrs_uz_kito_r.groupby(tikrinimas["klase"]).sum(),
    "T_pries_QRS_off": t_pries_qrs_pabaiga.groupby(tikrinimas["klase"]).sum(),
    "T_uz_kito_R": t_uz_kito_r.groupby(tikrinimas["klase"]).sum(),
    "P_on_po_QRS_on": p_po_qrs.groupby(tikrinimas["klase"]).sum(),
}))

print("\nPATIKSLINTA QRS LOGINĖ PATIKRA:")

# QRS_on ir QRS_off yra ECGPUWAVE anotacijos,
# todėl tikriname ECGPUWAVE aptiktą R_ecg, o ne ekspertinį R.
qrs_neapima_r_ecg = (
    tikrinimas["R_ecg"].notna()
    & tikrinimas["QRS_on"].notna()
    & tikrinimas["QRS_off"].notna()
    & (
        (tikrinimas["R_ecg"] < tikrinimas["QRS_on"])
        | (tikrinimas["R_ecg"] > tikrinimas["QRS_off"])
    )
)

print(
    "R_ecg nepatenka tarp QRS_on ir QRS_off:",
    qrs_neapima_r_ecg.sum()
)


# Kiek skiriasi ekspertinis R nuo ECGPUWAVE aptikto R
r_atstumas = (
    tikrinimas.loc[
        tikrinimas["R_ecg"].notna(),
        "R"
    ]
    - tikrinimas.loc[
        tikrinimas["R_ecg"].notna(),
        "R_ecg"
    ]
).abs()

print("\n|R - R_ecg| atstumas mėginiais:")
print(r_atstumas.describe().round(2))


print("\n|R - R_ecg| pagal klasę:")

tikrinimas["R_atstumas"] = (
    tikrinimas["R"] - tikrinimas["R_ecg"]
).abs()

print(
    tikrinimas
    .groupby("klase")["R_atstumas"]
    .agg(["count", "mean", "median", "max"])
    .round(2)
)


print("\nANKSTESNI 72 ATVEJAI - JŲ R ATSTUMAS:")

print(
    tikrinimas.loc[
        qrs_neapima_r,
        ["irasas", "klase", "R", "R_ecg",
         "R_atstumas", "QRS_on", "QRS_off"]
    ]
    .sort_values("R_atstumas", ascending=False)
    .to_string(index=False)
)

print("\nPIRMINIS DUOMENŲ TVARKYMAS:")

sutvarkyta = tiriamoji.copy()

# Aiškiai nelogiškos PR reikšmės:
# P pradžia negali būti ties QRS pradžia arba po jos.
blogas_pr = (
    sutvarkyta["PR"].notna()
    & (sutvarkyta["PR"] <= 0)
)

print("PR reikšmių, pakeičiamų į NaN:", blogas_pr.sum())

sutvarkyta.loc[blogas_pr, "PR"] = pd.NA

# Išsaugome atskirai – pradinės imties neperrašome.
sutvarkyta.to_csv(
    "duomenys/tiriamoji_sutvarkyta.csv",
    index=False
)

print("\nPAGRINDINIŲ POŽYMIŲ TRŪKSTAMUMAS PO TVARKYMO:")

PAGRINDINIAI = [
    "QRS",
    "RR_sant",
    "RR_pre",
    "QTc",
    "P_yra",
    "T_tipas"
]

truksta_pagrindiniu = pd.DataFrame({
    "kiekis": sutvarkyta[PAGRINDINIAI].isna().sum(),
    "procentai": sutvarkyta[PAGRINDINIAI].isna().mean() * 100
})

print(truksta_pagrindiniu.round(2))


print("\nEILUTĖS, KURIOSE TRŪKSTA BENT VIENO PAGRINDINIO POŽYMIO:")

truksta_bent_vieno = (
    sutvarkyta[PAGRINDINIAI]
    .isna()
    .any(axis=1)
)

print("Kiekis:", truksta_bent_vieno.sum())
print(
    "Procentai:",
    round(truksta_bent_vieno.mean() * 100, 2)
)


print("\nTRŪKSTAMOS EILUTĖS PAGAL KLASĘ:")

print(
    truksta_bent_vieno
    .groupby(sutvarkyta["klase"])
    .agg(["sum", "count", "mean"])
    .assign(procentai=lambda x: x["mean"] * 100)
    [["sum", "count", "procentai"]]
    .round(2)
)


print("\nPILNOS EILUTĖS PAGAL KLASĘ:")

pilna_eilute = ~truksta_bent_vieno

print(
    pilna_eilute
    .groupby(sutvarkyta["klase"])
    .sum()
)

print("\nPILNŲ EILUČIŲ AIBĖ PAGRINDINIŲ POŽYMIŲ ANALIZEI:")

analizei = (
    sutvarkyta
    .dropna(subset=PAGRINDINIAI)
    .copy()
)

print("Eilučių prieš:", len(sutvarkyta))
print("Eilučių po:", len(analizei))
print(
    "Pašalinta:",
    len(sutvarkyta) - len(analizei)
)

print("\nKLASIŲ PASISKIRSTYMAS PO TRŪKSTAMŲ EILUČIŲ PAŠALINIMO:")
print(analizei["klase"].value_counts().sort_index())

print("\nKLASIŲ PROPORCIJOS (%):")
print(
    (
        analizei["klase"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    ).round(2)
)

analizei.to_csv(
    "duomenys/tiriamoji_analizei.csv",
    index=False
)

print("\nAPRAŠOMOJI STATISTIKA PAGAL KLASĘ:")

SKAITINIAI_PAGRINDINIAI = [
    "QRS",
    "RR_sant",
    "RR_pre",
    "QTc"
]


def aprasomoji_statistika(x):
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


for col in SKAITINIAI_PAGRINDINIAI:
    print(f"\n{col}:")

    lentele = (
        analizei
        .groupby("klase")[col]
        .apply(aprasomoji_statistika)
        .unstack()
    )

    print(lentele.round(2))


print("\nP_YRA PASISKIRSTYMAS PAGAL KLASĘ (%):")

p_yra_proc = (
    pd.crosstab(
        analizei["klase"],
        analizei["P_yra"],
        normalize="index"
    ) * 100
)

print(p_yra_proc.round(1))


print("\nT_TIPAS PASISKIRSTYMAS PAGAL KLASĘ (%):")

t_tipas_proc = (
    pd.crosstab(
        analizei["klase"],
        analizei["T_tipas"],
        normalize="index"
    ) * 100
)

print(t_tipas_proc.round(1))

print("\nKURIAMOS PAGRINDINIŲ POŽYMIŲ VIZUALIZACIJOS:")

KLASIU_TVARKA = ["N", "L", "R", "V", "A"]

for col in ["QRS", "RR_sant", "RR_pre", "QTc"]:
    duomenys = [
        analizei.loc[analizei["klase"] == klase, col]
        for klase in KLASIU_TVARKA
    ]

    plt.figure(figsize=(8, 5))

    plt.boxplot(
        duomenys,
        tick_labels=KLASIU_TVARKA,
        showfliers=True
    )

    plt.title(f"{col} pasiskirstymas pagal klasę")
    plt.xlabel("Dūžio klasė")
    plt.ylabel(col)
    plt.tight_layout()

    plt.show()


# P_yra pasiskirstymas
p_yra_plot = (
    pd.crosstab(
        analizei["klase"],
        analizei["P_yra"],
        normalize="index"
    )
    .reindex(KLASIU_TVARKA)
    * 100
)

p_yra_plot.plot(
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
T_TIPAI = {
    0.0: "normali",
    1.0: "apversta",
    2.0: "tik teigiama",
    3.0: "tik neigiama",
    4.0: "dvifazė −/+",
    5.0: "dvifazė +/−"
}

t_tipas_plot = (
    pd.crosstab(
        analizei["klase"],
        analizei["T_tipas"],
        normalize="index"
    )
    .reindex(KLASIU_TVARKA)
    .rename(columns=T_TIPAI)
    * 100
)

t_tipas_plot.plot(
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
    analizei
    .groupby("klase")[SKAITINIAI_PAGRINDINIAI]
    .skew()
    .round(2)
)

print("\nMASTELIAI (min–max):")
print(
    analizei[SKAITINIAI_PAGRINDINIAI]
    .agg(["min", "max"])
    .round(2)
)

print("\nT_TIPAS KIEKIAI (retos kategorijos):")
print(
    pd.crosstab(
        analizei["klase"],
        analizei["T_tipas"]
    )
)


# Patikriname hipotezę, ar 232 įrašas lemia platesnį
# A klasės QRS pasiskirstymą.
yra232 = analizei["irasas"].astype(str) == "232"
a_kl = analizei["klase"] == "A"

print("\nA KLASĖ: 232 ĮRAŠAS (True) VS KITI (False), MEDIANOS:")
print(
    analizei[a_kl]
    .groupby(yra232[a_kl])[["QRS", "RR_post"]]
    .median()
    .round(1)
)


# Patikriname, iš kokių įrašų ateina labai ilgi
# R klasės RR_pre intervalai.
r_kl = analizei[analizei["klase"] == "R"]

print("\nR KLASĖS RR_pre > 2000 ms PAGAL ĮRAŠĄ:")
print(
    r_kl.loc[
        r_kl["RR_pre"] > 2000,
        "irasas"
    ].value_counts()
)


# Patikriname, ar N klasės RR_sant išskirtys
# dažniau susijusios su AFIB ritmu.
n_kl = analizei[analizei["klase"] == "N"]

q1 = n_kl["RR_sant"].quantile(0.25)
q3 = n_kl["RR_sant"].quantile(0.75)
iqr = q3 - q1

isk = ~n_kl["RR_sant"].between(
    q1 - 1.5 * iqr,
    q3 + 1.5 * iqr
)

afib = n_kl["ritmas"] == "(AFIB"

print(
    f"\nN RR_sant: AFIB dalis tarp išskirčių "
    f"{afib[isk].mean() * 100:.1f} %, "
    f"tarp kitų {afib[~isk].mean() * 100:.1f} %"
)