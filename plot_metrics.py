import matplotlib.pyplot as plt

# Data
models = ["Restormer", "NAFNet"]
psnr_values = [23.38, 14.64]
ssim_values = [0.4748, 0.3408]

# Plot
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# PSNR bar plot
ax[0].bar(models, psnr_values)
ax[0].set_title("PSNR Comparison")
ax[0].set_ylabel("PSNR (dB)")
ax[0].set_ylim(0, max(psnr_values) + 5)

# SSIM bar plot
ax[1].bar(models, ssim_values)
ax[1].set_title("SSIM Comparison")
ax[1].set_ylabel("SSIM")
ax[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig("metric_comparison.png")
plt.show()
print("✅ Graph saved as metric_comparison.png")



