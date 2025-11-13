import os
import sys
import torch

# -------------------------------------------------------------------------
# 🧩 Add submodule paths (Restormer and NAFNet)
# -------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESTORMER_PATH = os.path.join(ROOT_DIR, "Restormer")
NAFNET_PATH = os.path.join(ROOT_DIR, "NAFNet")

# Add both Restormer and NAFNet to sys.path (so Python can find them)
for path in [RESTORMER_PATH, NAFNET_PATH]:
    if path not in sys.path:
        sys.path.append(path)

# -------------------------------------------------------------------------
# 🧠 get_model(): Loads either Restormer or NAFNet with pretrained weights
# -------------------------------------------------------------------------
def get_model(model_name="restormer"):
    model_name = model_name.lower()
    model = None
    weights_path = None

    # ------------------------------------------------------------
    # 🚗 RESTORMER
    # ------------------------------------------------------------
    if model_name == "restormer":
        try:
            from basicsr.models.archs.restormer_arch import Restormer
        except Exception as e:
            print("❌ Failed to import Restormer architecture.")
            print("💡 Ensure file exists: Restormer/basicsr/models/archs/restormer_arch.py")
            print("Error details:", e)
            return None

        print("Loading Restormer architecture...")
        model = Restormer()
        weights_path = os.path.join(
            ROOT_DIR, "Restormer", "Denoising", "pretrained_models", "gaussian_color_denoising_sigma25.pth"
        )

    # ------------------------------------------------------------
    # 🚀 NAFNET
    # ------------------------------------------------------------
    elif model_name == "nafnet":
        # Step 1: Make sure the correct path exists
        nafnet_path = os.path.join(ROOT_DIR, "NAFNet")
        if nafnet_path not in sys.path:
            sys.path.append(nafnet_path)

        # Step 2: Import architecture
        try:
            from NAFNet.models.archs.NAFNet_arch import NAFNet
        except Exception as e:
            print("❌ Failed to import NAFNet architecture.")
            print("💡 Ensure file exists: NAFNet/models/archs/NAFNet_arch.py")
            print(f"Error details: {e}")
            return None

        print("Loading NAFNet architecture...")
        model = NAFNet(
            width=32,
            enc_blk_nums=[1, 1, 1, 28],
            middle_blk_num=1,
            dec_blk_nums=[1, 1, 1, 1]
        )

        weights_path = os.path.join(
            ROOT_DIR, "NAFNet", "pretrained_models", "NAFNet-SIDD-width32.pth"
        )

    else:
        print(f"❌ Unknown model name: {model_name}")
        return None

    # ------------------------------------------------------------
    # ⚙️ Load pretrained weights
    # ------------------------------------------------------------
    if weights_path and os.path.exists(weights_path):
        print(f"Loading pre-trained model weights from {weights_path}")
        try:
            state_dict = torch.load(weights_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"✅ Weights loaded successfully. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        except Exception as e:
            print(f"⚠️ Failed to load weights: {e}")
    else:
        print(f"⚠️ Warning: No weights found at {weights_path}. Using random initialization.")

    # ------------------------------------------------------------
    # 🧮 Move model to available device
    # ------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"⚙️ Moving {model_name.upper()} to {device} ...")
    model = model.to(device)

    return model
