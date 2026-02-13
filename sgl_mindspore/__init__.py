"""SGL-MindSpore package for SGLang with MindSpore."""

__version__ = "0.1.0"

# Import modules here for easier access
# from .module import *
from sgl_mindspore.utils import is_310p, patch_triton_310p, patch_memory_pool_310p

if is_310p():
    patch_triton_310p()
    patch_memory_pool_310p()
