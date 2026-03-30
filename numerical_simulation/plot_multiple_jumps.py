import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

L = 6
T = 200
# ---- file paths ----
file1 = Path().resolve().parent/f"Lindblad_simulation/numerical_simulation/multiple_jump_operator/data/lindblad_results_multiple_jumps_L{L}_T{T}_steps{T}.json"
file2 = Path().resolve().parent/f"Lindblad_simulation/numerical_simulation/lindbladian_simulation/single_jumps_data/lindblad_results_single_jumps_L{L}_T{T}_steps{T}.json"

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
plt.title(f"Energy vs Time for TFIM-{L}")
plt.legend()
plt.grid(True)
plt.tight_layout()
output_dir = Path().resolve().parent/f"Lindblad_simulation/plots"
energy_path = output_dir / f"Energy_L{L}_T{T}.png"
plt.savefig(energy_path, dpi=300)
print(f"Saved energy plot to {energy_path}")
plt.show()

# ---- Plot Overlap ----
plt.figure(figsize=(10, 6))
plt.plot(t1, p1, label="3 jump operators", linewidth=2)
plt.plot(t2, p2, label="Single jump operator", linestyle="--", linewidth=2)

plt.xlabel("Time")
plt.ylabel("<pGS>")
plt.title(f"Ground State Overlap vs Time for TFIM-{L}")
plt.legend()
plt.grid(True)
plt.tight_layout()
overlap_path = output_dir / f"Overlap_L{L}_T{T}.png"
plt.savefig(overlap_path, dpi=300)
print(f"Saved overlap plot to {overlap_path}")
plt.show()