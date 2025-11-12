import os
import torch
import argparse
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
from model import get_model

# -----------------------------------------------------------------------------
# 🧩 Helper: Load image and convert to torch tensor
# -----------------------------------------------------------------------------
def load_image(path):
    img = Image.open(path).convert("RGB")
    tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return tensor, img.size

# -----------------------------------------------------------------------------
# 🧩 Helper: Save torch tensor to image
# -----------------------------------------------------------------------------
def save_image(tensor, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tensor = tensor.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    img = Image.fromarray((tensor * 255).astype(np.uint8))
    img.save(path)

# -----------------------------------------------------------------------------
# 🚀 Inference Script
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run image denoising using NAFNet or Restormer.")
    parser.add_argument("--model", type=str, required=True, help="Model name: nafnet or restormer")
    parser.add_argument("--input", type=str, required=True, help="Path to input noisy image")
    parser.add_argument("--output", type=str, required=True, help="Path to save denoised output image")
    args = parser.parse_args()

    # --- Load model ---
    print(f"\n🚀 Running {args.model.upper()} on {args.input}")
    model = get_model(args.model)

    if model is None:
        print("❌ Model could not be loaded. Check model.py configuration.")
        exit()

    # --- Load input image ---
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        exit()

    img_tensor, img_size = load_image(args.input)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    img_tensor = img_tensor.to(device)

    # --- Run inference ---
    with torch.no_grad():
        output = model(img_tensor)

    # ✅ Normalize intensity safely
    output = output.clamp(0, 1)
    output = (output - output.min()) / (output.max() - output.min() + 1e-8)

    # ✅ Gentle color balance (prevent overexposure)
    mean_rgb = output.mean(dim=[0, 2, 3], keepdim=True)
    ref_mean = mean_rgb.mean()  # keep global luminance consistent
    output = output * (ref_mean / (mean_rgb + 1e-8))
    output = output.clamp(0, 1)

    # ✅ Optional: apply light gamma correction for natural tone
    output = torch.pow(output, 1 / 1.1)


    # --- Save result ---
    save_image(output, args.output)
    print(f"✅ Output saved to: {args.output}")
