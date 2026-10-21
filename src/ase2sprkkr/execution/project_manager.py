"""Compatibility import for the single execution manager.

Project membership is explicit on each RunRecord, not a bound manager state.
"""
from .manager import ExecutionManager

__all__ = ["ExecutionManager"]
