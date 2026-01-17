import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set Times New Roman font
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14

# ---------- Data: R^2 means ----------
data = [
    ("C1", "Random",  "Scratch", 0.68),
    ("C1", "Random",  "Finetune", 0.88),
    ("C1", "Distant", "Scratch", 0.21),
    ("C1", "Distant", "Finetune", 0.47),

    ("C2", "Random",  "Scratch", 0.92),
    ("C2", "Random",  "Finetune", 0.97),
    ("C2", "Distant", "Scratch", 0.62),
    ("C2", "Distant", "Finetune", 0.86),

    ("C3", "Random",  "Scratch", 0.84),
    ("C3", "Random",  "Finetune", 0.88),
    ("C3", "Distant", "Scratch", 0.44),
    ("C3", "Distant", "Finetune", 0.51),

    ("C4", "Random",  "Scratch", 0.85),
    ("C4", "Random",  "Finetune", 0.92),
    ("C4", "Distant", "Scratch", 0.59),
    ("C4", "Distant", "Finetune", 0.77),

    ("C5", "Random",  "Scratch", 0.82),
    ("C5", "Random",  "Finetune", 0.89),
    ("C5", "Distant", "Scratch", 0.28),
    ("C5", "Distant", "Finetune", 0.37),

    ("C6", "Random",  "Scratch", 0.78),
    ("C6", "Random",  "Finetune", 0.85),
    ("C6", "Distant", "Scratch", 0.48),
    ("C6", "Distant", "Finetune", 0.67),
]
df = pd.DataFrame(data, columns=["City", "Test set", "Training", "R2"])

# ---------- Safety checks ----------
cities = ["C1", "C2", "C3", "C4", "C5", "C6"]
expected_rows = len(cities) * 4
assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"
assert set(df["City"]) == set(cities), "Unexpected/missing cities"
assert set(df["Test set"]) == {"Random", "Distant"}, "Unexpected test-set labels"
assert set(df["Training"]) == {"Scratch", "Finetune"}, "Unexpected training labels"

# ---------- Prepare values ----------
x = np.arange(len(cities))
vals = {
    ("Random", "Scratch"):  df[(df["Test set"]=="Random")  & (df["Training"]=="Scratch")].set_index("City").loc[cities]["R2"].to_numpy(),
    ("Random", "Finetune"): df[(df["Test set"]=="Random")  & (df["Training"]=="Finetune")].set_index("City").loc[cities]["R2"].to_numpy(),
    ("Distant","Scratch"):  df[(df["Test set"]=="Distant") & (df["Training"]=="Scratch")].set_index("City").loc[cities]["R2"].to_numpy(),
    ("Distant","Finetune"): df[(df["Test set"]=="Distant") & (df["Training"]=="Finetune")].set_index("City").loc[cities]["R2"].to_numpy(),
}

# Color encodes test set; hatch encodes training
colors  = {"Random": "tab:blue", "Distant": "tab:orange"}
hatches = {"Scratch": "//", "Finetune": ""}

width = 0.20
offsets = {
    ("Random","Scratch"):  -1.5*width,
    ("Random","Finetune"): -0.5*width,
    ("Distant","Scratch"):  0.5*width,
    ("Distant","Finetune"): 1.5*width,
}

# ---------- Plot ----------
fig = plt.figure(figsize=(10.5, 4.8))
for (test, train), off in offsets.items():
    plt.bar(
        x + off,
        vals[(test, train)],
        width=width,
        color=colors[test],
        hatch=hatches[train],
        edgecolor="black",
        linewidth=0.6,
        label=f"{test} – {train}",
    )

plt.xticks(x, cities)
plt.ylim(0, 1.15)
plt.ylabel(r"$R^2$")
plt.legend(ncols=2, frameon=False)
plt.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6)
plt.tight_layout()

# ---------- Save to results folder in same directory as this script ----------
script_dir = Path(__file__).parent
out_dir = script_dir / "results"
out_dir.mkdir(parents=True, exist_ok=True)

png_path = out_dir / "r2_random_vs_distant.png"
pdf_path = out_dir / "r2_random_vs_distant.pdf"

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")

plt.show()

print(f"Saved: {png_path.resolve()}")
print(f"Saved: {pdf_path.resolve()}")
