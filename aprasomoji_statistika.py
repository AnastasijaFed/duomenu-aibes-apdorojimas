from pathlib import Path
import pandas as pd

df = pd.read_csv("duomenys/tiriamoji.csv")
output_dir = Path("duomenys/")
output_dir.mkdir(parents=True, exist_ok=True)

quantitative = ["RR_pre", "RR_post", "RR_vid", "RR_sant_vid", "RR_sant_post",
                "PR", "QRS", "QT", "QTc",
                "P_amp", "T_amp", "ST", "QRS_max", "QRS_min"]

categorical = ["P_yra", "T_tipas"]


def fmt(value):
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


num = df[quantitative]

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


cat = df[categorical]

cat_stats = pd.DataFrame("–", index=categorical, columns=num_stats.columns)
cat_stats["Neegzistuojančios reikšmės"] = cat.isna().sum()


table = pd.concat([num_stats, cat_stats])
table.index.name = "Požymis"

print(table.to_string())

for col in categorical:
    print(f"\n{col} counts:")
    print(df[col].value_counts(dropna=False))

table.to_csv(output_dir / "aprasomoji_statistika.csv", encoding="utf-8-sig")