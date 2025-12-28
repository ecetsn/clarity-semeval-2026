from typing import Dict

from .mlp_projection import ProjectionMLP
from .base_projection import BaseProjection


def build_projection(proj_cfg: Dict) -> BaseProjection:
    """
    Factory method for projection layers.
    """

    proj_type = proj_cfg.get("type")
    params = proj_cfg.get("params", {})

    if proj_type == "mlp":
        return ProjectionMLP(**params)

    elif proj_type == "identity":
        from .identity_projection import IdentityProjection
        return IdentityProjection(**params)

    else:
        raise ValueError(f"Unknown projection type: {proj_type}")
