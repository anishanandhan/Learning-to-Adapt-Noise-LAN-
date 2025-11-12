import matplotlib.pyplot as plt
from PIL import Image
import os

# --- Paths (change if needed) ---
GT_PATH = "polyu/gt/SonyA7II_plant_0007.png"
RESTORMER_PATH = "output_restormer/SonyA7II_plant_0007.png"
NAFNET_PATH = "output_nafnet/SonyA7II_plant_0007.png"

# --- Check all paths exist ---
for p in [GT_PATH, RESTORMER_PATH, NAFNET_PATH]:
    if not os.path.exists(p):
        print(f"❌ Missing file: {p}")
        exit()

# --- Load images ---
gt = Image.open(GT_PATH).convert("RGB")
restormer = Image.open(RESTORMER_PATH).convert("RGB")
nafnet = Image.open(NAFNET_PATH).convert("RGB")

# --- Plot the 3 images side by side ---
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(gt)
axs[0].set_title("Ground Truth (GT)")
axs[0].axis("off")

axs[1].imshow(restormer)
axs[1].set_title("Restormer (Base Paper)")
axs[1].axis("off")

axs[2].imshow(nafnet)
axs[2].set_title("Proposed NAFNet")
axs[2].axis("off")

plt.tight_layout()
plt.savefig("comparison_grid.png", dpi=300)
plt.show()
print("✅ Saved visual comparison as comparison_grid.png")


