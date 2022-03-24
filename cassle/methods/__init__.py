from cassle.methods.barlow_twins import BarlowTwins
from cassle.methods.base import BaseModel
from cassle.methods.byol import BYOL
from cassle.methods.deepclusterv2 import DeepClusterV2
from cassle.methods.dino import DINO
from cassle.methods.linear import LinearModel
from cassle.methods.mocov2plus import MoCoV2Plus
from cassle.methods.nnclr import NNCLR
from cassle.methods.ressl import ReSSL
from cassle.methods.simclr import SimCLR
from cassle.methods.simsiam import SimSiam
from cassle.methods.swav import SwAV
from cassle.methods.vicreg import VICReg
from cassle.methods.wmse import WMSE

METHODS = {
    # base classes
    "base": BaseModel,
    "linear": LinearModel,
    # methods
    "barlow_twins": BarlowTwins,
    "byol": BYOL,
    "deepclusterv2": DeepClusterV2,
    "dino": DINO,
    "mocov2plus": MoCoV2Plus,
    "nnclr": NNCLR,
    "ressl": ReSSL,
    "simclr": SimCLR,
    "simsiam": SimSiam,
    "swav": SwAV,
    "vicreg": VICReg,
    "wmse": WMSE,
}
__all__ = [
    "BarlowTwins",
    "BYOL",
    "BaseModel",
    "DeepClusterV2",
    "DINO",
    "LinearModel",
    "MoCoV2Plus",
    "NNCLR",
    "ReSSL",
    "SimCLR",
    "SimSiam",
    "SwAV",
    "VICReg",
    "WMSE",
]

try:
    from cassle.methods import dali  # noqa: F401
except ImportError:
    pass
else:
    __all__.append("dali")
