from pathlib import Path
import pandas as pd

df = pd.read_csv("duomenys/tiriamoji.csv")
output_dir = Path("duomenys/apras_stat_klases")
output_dir.mkdir(parents=True, exist_ok=True)

quantitative = ["RR_pre", "RR_post", "RR_vid", "RR_sant_vid", "RR_sant_post", "PR", "QRS", "QT", "QTc","P_amp", "T_amp", "ST", "QRS_max", "QRS_min"]
categorical = ["P_yra", "T_tipas"]
classes = ["N", "L", "R", "V", "A"]


def fmt(value):
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def describe(data):
    num = data[quantitative]

    num_stats = pd.DataFrame({
        "Minimumas": num.min(),
        "1-as kvartilis": num.quantile(0.25),
        "Mediana": num.median(),
        "Vidurkis": num.mean(),
        "3-as kvartilis": num.quantile(0.75),
        "Maksimumas": num.max(),
        "Dispersija": num.var(),
    }).map(fmt)

    num_stats["Neegzistuojančios reikšmės"] = num.isna().sum()

    cat_stats = pd.DataFrame("–", index=categorical, columns=num_stats.columns)
    cat_stats["Neegzistuojančios reikšmės"] = data[categorical].isna().sum()

    table = pd.concat([num_stats, cat_stats])
    table.index.name = "Požymis"
    return table


for cls in classes:
    class_data = df[df["klase"] == cls]
    table = describe(class_data)

    print(f"\nKlasė {cls} (n = {len(class_data)})\n")
    print(table.to_string())

    table.to_csv(output_dir / f"{cls}_klase_AP.csv", encoding="utf-8-sig")