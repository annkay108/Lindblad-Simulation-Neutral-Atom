import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---- file paths ----
file1 = Path().resolve().parent/"Lindblad_simulation/numerical_simulation/multiple_jump_operator/data/lindblad_results_multiple_jumps_L6_T200_steps200.json"
file2 = Path().resolve().parent/"Lindblad_simulation/numerical_simulation/lindbladian_simulation/single_jumps_data/lindblad_results_single_jumps_L6_T200_steps200.json"

# ---- load JSON ----
def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    return (
        np.array(data["time_series"]),
        np.array(data["avg_energy"]),
        np.array(data["avg_pGS"]),
    )

t1, E1, p1 = load_data(file1)
t2, E2, p2 = load_data(file2)

# ---- Plot Energy ----
plt.figure(figsize=(10, 6))
plt.plot(t1, E1, label="3 jump operators", linewidth=2)
plt.plot(t2, E2, label="Single jump operator", linestyle="--", linewidth=2)

plt.xlabel("Time")
plt.ylabel("<E>")
plt.title("Energy vs Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- Plot Overlap ----
plt.figure(figsize=(10, 6))
plt.plot(t1, p1, label="3 jump operators", linewidth=2)
plt.plot(t2, p2, label="Single jump operator", linestyle="--", linewidth=2)

plt.xlabel("Time")
plt.ylabel("<pGS>")
plt.title("Ground State Overlap vs Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()