import os
import torch
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
#  Configuration
# ==========================================================
GT_DIR = "polyu/gt"                      # Ground truth images
RESTORMER_DIR = "output_restormer"       # LAN + Restormer outputs
NAFNET_DIR = "output_nafnet"             # LAN + NAFNet outputs
RESULT_CSV = "results/metrics_comparison.csv"
RESULT_PLOT = "results/psnr_ssim_comparison.png"

os.makedirs("results", exist_ok=True)

# ==========================================================
#  Helper Functions
# ==========================================================
def load_image(path):
    """Load image and normalize to [0,1] range."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0

def evaluate_pair(pred_path, gt_path):
    """Compute PSNR and SSIM between prediction and GT."""
    pred = load_image(pred_path)
    gt = load_image(gt_path)
    psnr_val = psnr(gt, pred, data_range=1.0)
    ssim_val = ssim(gt, pred, channel_axis=2, data_range=1.0)
    return psnr_val, ssim_val

# ==========================================================
#  Evaluation
# ==========================================================
results = []
files = sorted(os.listdir(GT_DIR))

print(f"🔍 Found {len(files)} images for evaluation.\n")

for filename in files:
    gt_path = os.path.join(GT_DIR, filename)
    restormer_path = os.path.join(RESTORMER_DIR, filename)
    nafnet_path = os.path.join(NAFNET_DIR, filename)

    if not os.path.exists(restormer_path) or not os.path.exists(nafnet_path):
        print(f"⚠️ Skipping {filename}: missing output.")
        continue

    psnr_r, ssim_r = evaluate_pair(restormer_path, gt_path)
    psnr_n, ssim_n = evaluate_pair(nafnet_path, gt_path)

    results.append({
        "Image": filename,
        "PSNR_Restormer": psnr_r,
        "SSIM_Restormer": ssim_r,
        "PSNR_NAFNet": psnr_n,
        "SSIM_NAFNet": ssim_n,
        "ΔPSNR": psnr_n - psnr_r,
        "ΔSSIM": ssim_n - ssim_r
    })

# ==========================================================
#  Save & Display Results
# ==========================================================
if not results:
    print("❌ No results computed — check your folder paths.")
    exit()

df = pd.DataFrame(results)
df.loc["Average"] = df.mean(numeric_only=True)
df.to_csv(RESULT_CSV, index=False)
print(f"\n✅ Metrics saved to: {RESULT_CSV}\n")
print(df.tail())

# ==========================================================
#  Visualization
# ==========================================================
avg_psnr_r = df["PSNR_Restormer"].mean()
avg_psnr_n = df["PSNR_NAFNet"].mean()
avg_ssim_r = df["SSIM_Restormer"].mean()
avg_ssim_n = df["SSIM_NAFNet"].mean()

plt.figure(figsize=(6, 4))
x = np.arange(2)
plt.bar(x - 0.15, [avg_psnr_r, avg_ssim_r], 0.3, label="Restormer")
plt.bar(x + 0.15, [avg_psnr_n, avg_ssim_n], 0.3, label="NAFNet")
plt.xticks(x, ["PSNR", "SSIM"])
plt.title("Average Performance Comparison (LAN Adaptation)")
plt.ylabel("Metric Value")
plt.legend()
plt.tight_layout()
plt.savefig(RESULT_PLOT)
print(f"📊 Comparison plot saved to: {RESULT_PLOT}")
plt.show()
