"""Deprecated taxonomy enums preserved for backward compatibility.

SkillVisibility has been moved to skilleval.core.types.
CompositionPattern was removed during the Phase A taxonomy cleanup.
"""

from enum import Enum

# Re-export from canonical location for backward compatibility.
from skilleval.core.types import SkillVisibility  # noqa: F401


class CompositionPattern(str, Enum):
    """DEPRECATED: Was Category 1 in original code, removed to align with paper."""
    STRAIGHT_LINE = "SL"
    PICK_ONE = "PO"
    FIX_UNTIL_PASS = "FP"
