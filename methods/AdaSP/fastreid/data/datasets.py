# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""


from ..utils.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for datasets
It must returns an instance of :class:`Backbone`.
"""

# Person re-id datasets
from .cuhk03 import CUHK03
from .dukemtmcreid import DukeMTMC
from .market1501 import Market1501
from .msmt17 import MSMT17

# Vehicle re-id datasets
from .veri import VeRi
from .vehicleid import VehicleID, SmallVehicleID, MediumVehicleID, LargeVehicleID
from .veriwild import VeRiWild, SmallVeRiWild, MediumVeRiWild, LargeVeRiWild

from .vru import VRU
from .vru_fog import VRU_Fog
from .vru_rain import VRU_Rain

from .uav_veid import UAV_VeID
from .uav_veid_fog import UAV_VeID_Fog
from .uav_veid_rain import UAV_VeID_Rain

__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]
