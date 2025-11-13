import sys, os
# Ensure local basicsr_NAFNET acts like basicsr
sys.modules['basicsr'] = sys.modules.get('basicsr_NAFNET', None)

# Try importing utils from local folder
try:
    from basicsr_NAFNET.utils import get_root_logger, scandir
except ImportError:
    # Fallback in case the utils path is different
    from ..utils import get_root_logger, scandir
