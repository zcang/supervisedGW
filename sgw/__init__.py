# __init__ file
import warnings
warnings.filterwarnings('ignore')

from . import sgw
from . import utils

from .sgw import (supervised_gromov_wasserstein)
from .utils import (recover_full_coupling)

__version__="0.0.1"

__all__ = ['supervised_gromov_wasserstein', 'recover_full_coupling']