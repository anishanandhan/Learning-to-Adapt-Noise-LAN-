# ===============================================================
# Minimal Local_Base definition for NAFNet compatibility on macOS
# ===============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class Local_Base(nn.Module):
    def __init__(self):
        super(Local_Base, self).__init__()

    def convert(self, *args, **kwargs):
        """
        Dummy method to avoid import issues on MPS (Mac Metal).
        Used by NAFNetLocal.
        """
        pass
