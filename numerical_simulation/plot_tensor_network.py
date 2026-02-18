import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_FILE = "data/mps_gate_implementation_results_news_9sites_250iter1measure.json"
SAVE_PLOTS = True
PLOT_DIR = "plots"
# TRUE_ENERGY = -5.4315195827374945
# TRUE_ENERGY = -6.85018777
TRUE_ENERGY = -12.52776194

os.makedirs(PLOT_DIR, exist_ok=True)

with open(DATA_FILE, "r") as f:
    results = json.load(f)

df = pd.DataFrame(results)

def save(fig, name):
    if SAVE_PLOTS:
        path = os.path.join(PLOT_DIR, name)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {path}")

# --------------------------------------
# Plot 1: Energy vs Iterations
# --------------------------------------

fig = plt.figure(figsize=(8,6))
sns.lineplot(data=df, x="iterations", y="energy", hue="max_bond_dim",    style="max_bond_dim",
    markers=True,
    dashes=False,
    palette="tab10")
plt.axhline(TRUE_ENERGY, color="red", linestyle="--", label="True Ground State")
plt.title("Energy vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("Energy")
plt.grid(True)
plt.legend()
save(fig, "energy_vs_iterations1_9sites.png")
plt.show()


