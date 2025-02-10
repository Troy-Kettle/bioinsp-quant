"""
Bioinsp-quant 
--------------------------

Bioinspired machine learning algorithms for quantitative finance.

"""

# Package metadata
__version__ = "0.1.0"

# Set up package-level logging
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("my_library package initialized.")

# Re-export functions from submodules to simplify the public API
# from .module_a import function_a

# Define the public API of the package
# __all__ = ['function_b']

