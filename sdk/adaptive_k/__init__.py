"""
Adaptive-K: Entropy-guided dynamic expert selection for MoE models.

Reduce inference costs by 30-50% with proven methodology.

License: Apache 2.0 (Community) or Commercial License
Website: https://adaptive-k.vertexdata.it
Contact: amministrazione@vertexdata.it
"""

__version__ = "0.1.2"
__author__ = "Vertex Data"
__email__ = "amministrazione@vertexdata.it"

from .router import AdaptiveKRouter
from .calibration import Calibrator
from .licensing import LicenseValidator, LicenseInfo, print_license_info

__all__ = [
    "AdaptiveKRouter", 
    "Calibrator",
    "LicenseValidator",
    "LicenseInfo",
    "print_license_info"
]
