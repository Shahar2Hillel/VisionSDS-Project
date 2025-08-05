import pandas as pd
import matplotlib.pyplot as plt
import os

# === CONFIG ===
OUTPUT_DIR = "../output_all_tools/output_vis"
CSV_PATH = os.path.join(OUTPUT_DIR, "wrong_keypoints.csv")
SORT_BARS = True  # sort by most frequent errors

# === LOAD CSV ===
df = pd.read_csv(CSV_PATH)

# === ERROR COUNT ===
error_counts = df.groupby("keypoint_name").size()

# === AVERAGE PIXEL DISTANCE ===
avg_pixel_error = df.groupby("keypoint_name")["pixel_dist"].mean()

# === SORT (optional) ===
if SORT_BARS:
    sort_order = error_counts.sort_values(ascending=False).index
    error_counts = error_counts.loc[sort_order]
    avg_pixel_error = avg_pixel_error.loc[sort_order]

# === PLOT ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot error counts
bars1 = axes[0].bar(error_counts.index, error_counts.values, color="red", alpha=0.7)
axes[0].set_title("Wrong Keypoints Count", fontsize=14)
axes[0].set_xlabel("Keypoint Name", fontsize=12)
axes[0].set_ylabel("Error Count", fontsize=12)
for bar in bars1:
    yval = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval),
                 ha='center', va='bottom', fontsize=10)

# Plot average pixel distance
bars2 = axes[1].bar(avg_pixel_error.index, avg_pixel_error.values, color="blue", alpha=0.7)
axes[1].set_title("Average Pixel Error per Keypoint", fontsize=14)
axes[1].set_xlabel("Keypoint Name", fontsize=12)
axes[1].set_ylabel("Average Pixel Distance", fontsize=12)
for bar in bars2:
    yval = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.1f}",
                 ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "keypoint_error_analysis.png"), dpi=300)
