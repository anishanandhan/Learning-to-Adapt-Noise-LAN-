import os
import sys
import torch
import argparse

# -----------------------------------------------------------------------------
# 🧭 Ensure Python can find local modules (NAFNet + Restormer)
# -----------------------------------------------------------------------------
root_dir = os.path.dirname(os.path.abspath(__file__))
nafnet_path = os.path.join(root_dir, "NAFNet")
restormer_path = os.path.join(root_dir, "Restormer")

for p in [root_dir, nafnet_path, restormer_path,
          os.path.join(nafnet_path, "models", "archs"),
          os.path.join(restormer_path, "basicsr", "models", "archs")]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.append(p)


# -----------------------------------------------------------------------------
# 🧠 get_model(): Load either Restormer or NAFNet
# -----------------------------------------------------------------------------
def get_model(model_name="restormer"):
    """
    Loads the specified model (Restormer or NAFNet)
    and initializes pretrained weights if available.
    """

    model = None
    weights_path = None

    # -------------------------------------------------------------------------
    # 🧩 NAFNet
    # -------------------------------------------------------------------------
    if model_name.lower() == "nafnet":
        try:
            from NAFNet.models.archs.NAFNet_arch import NAFNet
        except Exception as e:
            print("⚠️ Failed to import NAFNet architecture.")
            print("💡 Ensure file exists: NAFNet/models/archs/NAFNet_arch.py")
            print(f"Error details: {e}")
            return None

        print("Loading NAFNet architecture...")
        model = NAFNet(width=32, enc_blk_nums=[1, 1, 1, 28],
                       middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1])
        weights_path = os.path.join(root_dir, "NAFNet", "pretrained_models", "NAFNet-SIDD-width32.pth")

    # -------------------------------------------------------------------------
    # 🧩 Restormer
    # -------------------------------------------------------------------------
    elif model_name.lower() == "restormer":
        try:
            from Restormer.basicsr.models.archs.restormer_arch import Restormer
        except Exception as e:
            print("⚠️ Failed to import Restormer architecture.")
            print("💡 Ensure file exists: Restormer/basicsr/models/archs/restormer_arch.py")
            print(f"Error details: {e}")
            return None

        print("Loading Restormer architecture...")
        model = Restormer()
        weights_path = os.path.join(root_dir, "Restormer", "Denoising",
                                    "pretrained_models", "gaussian_color_denoising_sigma25.pth")

    else:
        raise ValueError("❌ Unknown model name. Use 'restormer' or 'nafnet'.")

    # -------------------------------------------------------------------------
    # ⚙️ Load pretrained weights
    # -------------------------------------------------------------------------
    if weights_path and os.path.exists(weights_path):
        print(f"Loading pre-trained model weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"✅ Weights loaded successfully. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    else:
        print(f"⚠️ Warning: Weights not found at {weights_path}. Using random initialization.")

    return model


# -----------------------------------------------------------------------------
# 🚀 Main entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LAN Image Restoration Models")
    parser.add_argument("--model", type=str, default="nafnet",
                        help="Model to use: 'nafnet' or 'restormer'")
    parser.add_argument("--method", type=str, default="lan",
                        help="Restoration method (e.g., lan, nbr2nbr, etc.)")
    parser.add_argument("--self_loss", type=str, default="nbr2nbr",
                        help="Type of self-supervised loss")
    parser.add_argument("--loops", type=int, default=5,
                        help="Number of iterations to run")
    args = parser.parse_args()

    print(f"\n🚀 Running model: {args.model.upper()} | Method: {args.method} | Self-loss: {args.self_loss} | Loops: {args.loops}\n")

    model = get_model(args.model)

    if model is None:
        print("❌ Model failed to load. Exiting.")
        sys.exit(1)

    # Detect available device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"✅ Using device: {device}")
    model = model.to(device)

    print("\n✅ Model initialized successfully!")
 