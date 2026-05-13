import logging
from ._logging import configure_logger
from . import data
from . import module
from . import plot
from ._constants import *

from .module.registration import run_registration
from .module.transform import apply_sitk_transform, apply_affine_transform

log = logging.getLogger("xenium_align")
configure_logger(log)