"""
examples/mock_gbm_hmm.py

Mock Glioblastoma HMM Example
-----------------------------
Small synthetic dataset + HMM training + Viterbi decoding.

This script is meant as an example for the repository:
    glioblastoma-hmm-progression (or similar)

Requirements:
    pip install numpy pandas hmmlearn matplotlib
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt


# ----------------------------
# 1. Create a small mock GBM dataset
# ----------------------------
# We simulate 3 patients, each with 5 timepoints.
#
# Features:
#   - tumor_volume (cm^3): approximate total tumor volume
#   - enhancing_volume (cm^3): contrast-enhancing tumor volume
#
# Patterns:
#   - Patient 1: mostly stable / responding
#   - Patient 2: clear progression (growth over time)
#   - Patient 3: pseudoprogression-like (up then down)

data_rows = [
    # patient_id, timepoint, tumor_volume, enhancing_volume

    # Patient 1: mostly stable / responding (around 20–25 cm^3)
    ("P1", 0,  22.0,  8.0),
    ("P1", 1,  23.0,  8.1),
    ("P1", 2,  21.5,  7.8),
    ("P1", 3,  22.3,  7.9),
    ("P1", 4,  21.8,  7.7),

    # Patient 2: stable, then strong progression (up to ~70 cm^3)
    ("P2", 0,  28.0,  10.0),
    ("P2", 1,  30.0,  11.0),
    ("P2", 2,  40.0,  16.0),  # growth
    ("P2", 3,  55.0,  23.0),  # more growth
    ("P2", 4,  70.0,  30.0),

    # Patient 3: pseudoprogression-like (increase then partial regression)
    ("P3", 0,  24.0,   9.0),
    ("P3", 1,  32.0,  13.0),
    ("P3", 2,  38.0,  16.0),
    ("P3", 3,  30.0,  11.5),  # decreases
    ("P3", 4,  27.0,  10.0),
]

df = pd.DataFrame(
    data_rows,
    columns=["patient_id", "timepoint", "tumor_volume", "enhancing_volume"]
)

print("Mock GBM data:")
print(df)
print("\n")


# ----------------------------
# 2. Prepare data for HMM
# ----------------------------

# Sort by patient + timepoint to be safe
df = df.sort_values(["patient_id", "timepoint"])

# Feature matrix (n_samples_total, n_features)
X = df[["tumor_volume", "enhancing_volume"]].values

# Sequence lengths per patient (required by hmmlearn)
lengths = (
    df.groupby("patient_id")
      .size()
      .tolist()
)

patient_ids = df["patient_id"].unique().tolist()

print("Feature matrix shape:", X.shape)
print("Sequence lengths:", lengths)
print("Patient IDs:", patient_ids)
print("\n")


# ----------------------------
# 3. Fit a 3-state Gaussian HMM
# ----------------------------

n_states = 3  # e.g., Stable / Indeterminate / Progression

hmm = GaussianHMM(
    n_components=n_states,
    covariance_type="full",
    n_iter=200,
    random_state=42
)

hmm.fit(X, lengths)

print("Fitted HMM parameters:")
print("Start probabilities:", hmm.startprob_)
print("Transition matrix:\n", hmm.transmat_)
print("\nMeans of each state (tumor_volume, enhancing_volume):")
print(hmm.means_)
print("\n")


# ----------------------------
# 4. Decode hidden states (Viterbi)
# ----------------------------

# Predict the most likely state for each timepoint
hidden_states = hmm.predict(X, lengths=lengths)

# Add decoded states back to dataframe
df["hidden_state"] = hidden_states

print("Decoded hidden states per row:")
print(df)
print("\n")

# Also print per-patient trajectories (nice for README/demo)
for pid in patient_ids:
    sub = df[df["patient_id"] == pid].copy()
    print(f"Patient {pid} trajectory:")
    print(sub[["timepoint", "tumor_volume", "enhancing_volume", "hidden_state"]])
    print()


# ----------------------------
# 5. Simple visualization
# ----------------------------

# Map states to colors/labels for plotting
state_colors = {
    0: "C0",  # e.g., Stable
    1: "C1",  # e.g., Indeterminate / Pseudoprogression
    2: "C2",  # e.g., True Progression
}

state_labels = {
    0: "State 0",
    1: "State 1",
    2: "State 2",
}

fig, axes = plt.subplots(len(patient_ids), 1, figsize=(8, 8), sharex=True)

if len(patient_ids) == 1:
    axes = [axes]  # ensure iterable

for ax, pid in zip(axes, patient_ids):
    sub = df[df["patient_id"] == pid]
    ax.set_title(f"Patient {pid} – Tumor Volume and Inferred States")
    ax.set_ylabel("Tumor Volume (cm³)")

    # Plot tumor volume over time
    ax.plot(sub["timepoint"], sub["tumor_volume"], marker="o")

    # Overlay hidden states as colored markers
    for _, row in sub.iterrows():
        state = int(row["hidden_state"])
        ax.scatter(
            row["timepoint"],
            row["tumor_volume"],
            color=state_colors[state],
            label=state_labels[state]
        )

    # Avoid duplicated labels in the legend
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper left")

axes[-1].set_xlabel("Timepoint")
plt.tight_layout()
plt.show()
