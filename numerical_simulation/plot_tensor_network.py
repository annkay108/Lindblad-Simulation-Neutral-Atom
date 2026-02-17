import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_FILE = "data/mps_gate_implementation_results_news_8sites1.json"
SAVE_PLOTS = True
PLOT_DIR = "plots"
# TRUE_ENERGY = -5.4315195827374945
TRUE_ENERGY = -11.10821467

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
save(fig, "energy_vs_iterations1_6sites.png")
plt.show()

# --------------------------------------
# Plot 2: Runtime vs Iterations
# --------------------------------------

fig = plt.figure(figsize=(8,6))
sns.lineplot(data=df, x="iterations", y="execution_time", hue="max_bond_dim", marker="o")
plt.title("Execution Time vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("Execution Time (seconds)")
plt.grid(True)
save(fig, "runtime_vs_iterations1.png")
plt.show()

# --------------------------------------
# Plot 3: Runtime vs Bond Dimension
# --------------------------------------

fig = plt.figure(figsize=(8,6))
sns.lineplot(data=df, x="max_bond_dim", y="execution_time", hue="iterations", marker="o")
plt.title("Execution Time vs Max Bond Dimension")
plt.xlabel("Max Bond Dimension")
plt.ylabel("Execution Time (seconds)")
plt.grid(True)
save(fig, "runtime_vs_bond_dim1.png")
plt.show()

# --------------------------------------
# Plot 4: Energy vs Bond Dimension
# --------------------------------------

fig = plt.figure(figsize=(8,6))
sns.lineplot(data=df, x="max_bond_dim", y="energy", hue="iterations", marker="o")
plt.axhline(TRUE_ENERGY, color="red", linestyle="--", label="True Ground State")
plt.title("Energy vs Max Bond Dimension")
plt.xlabel("Max Bond Dimension")
plt.ylabel("Energy")
plt.grid(True)
plt.legend()
save(fig, "energy_vs_bond_dim1.png")
plt.show()

# --------------------------------------
# Plot 5: Heatmap (Runtime)
# --------------------------------------

fig = plt.figure(figsize=(10,6))
pivot = df.pivot(index="iterations", columns="max_bond_dim", values="execution_time")
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="rocket_r")
plt.title("Runtime Heatmap (seconds)")
plt.xlabel("Max Bond Dimension")
plt.ylabel("Iterations")
save(fig, "runtime_heatmap1.png")
plt.show()

print("\nAll plots generated successfully.")
if SAVE_PLOTS:
    print(f"Plots saved to folder: {PLOT_DIR}")
