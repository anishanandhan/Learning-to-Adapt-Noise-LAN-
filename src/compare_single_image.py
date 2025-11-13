import os
import torch
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 📸 CONFIGURATION
# -----------------------------------------------------------------------------
GT_PATH = "polyu/gt/SonyA7II_plant_0007.png"
RESTORMER_PATH = "output_restormer/SonyA7II_plant_0007.png"
NAFNET_PATH = "output_nafnet/SonyA7II_plant_0007.png"
SAVE_COMPARE_IMAGE = "comparison_result.png"

# -----------------------------------------------------------------------------
# 🧠 Helper Functions
# -----------------------------------------------------------------------------
def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img) / 255.0

def compute_metrics(gt, pred):
    return psnr(gt, pred, data_range=1.0), ssim(gt, pred, channel_axis=2, data_range=1.0)

# -----------------------------------------------------------------------------
# 🧮 Compute PSNR and SSIM
# -----------------------------------------------------------------------------
gt = load_image(GT_PATH)
rest = load_image(RESTORMER_PATH)
naf = load_image(NAFNET_PATH)

psnr_rest, ssim_rest = compute_metrics(gt, rest)
psnr_naf, ssim_naf = compute_metrics(gt, naf)

# -----------------------------------------------------------------------------
# 📊 Display Results
# -----------------------------------------------------------------------------
print("\n🔍 **Image Quality Comparison (SonyA7II_plant_0007.png)**")
print("----------------------------------------------------------")
print(f"Restormer → PSNR: {psnr_rest:.2f} dB | SSIM: {ssim_rest:.4f}")
print(f"NAFNet    → PSNR: {psnr_naf:.2f} dB | SSIM: {ssim_naf:.4f}")

# -----------------------------------------------------------------------------
# 🖼️ Visualization: GT vs Restormer vs NAFNet
# -----------------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(gt)
axs[0].set_title("Ground Truth")
axs[1].imshow(rest)
axs[1].set_title(f"Restormer\nPSNR={psnr_rest:.2f} | SSIM={ssim_rest:.3f}")
axs[2].imshow(naf)
axs[2].set_title(f"NAFNet\nPSNR={psnr_naf:.2f} | SSIM={ssim_naf:.3f}")

for ax in axs:
    ax.axis('off')

plt.tight_layout()
plt.savefig(SAVE_COMPARE_IMAGE)
plt.show()

print(f"\n✅ Comparison image saved as: {SAVE_COMPARE_IMAGE}")
