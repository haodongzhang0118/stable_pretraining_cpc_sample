"""Core model components for SSL whales project."""

from .backbone import cpc_backbone
from .forward import cpc_forward
from .loss import cpc_loss

__all__ = [
    "cpc_backbone",
    "cpc_forward", 
    "cpc_loss",
]

