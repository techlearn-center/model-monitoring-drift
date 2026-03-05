"""
Retraining Module

Provides automated retraining triggers and pipeline management
when drift or performance degradation is detected.
"""

from .trigger import RetrainingTrigger

__all__ = ["RetrainingTrigger"]
