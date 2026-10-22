"""Read-only adapters around released graph-prompt implementations."""

from .gretpool import GRetPoolAdapter, ProjectorCapture

__all__ = ["GRetPoolAdapter", "ProjectorCapture"]
