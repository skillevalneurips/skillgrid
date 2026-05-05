"""Non-invasive debug helpers. Wrap a model with DebugModel to capture
every ``generate`` / ``generate_with_tools`` call to a jsonl file.

Removing the wrapper restores original behavior byte-for-byte — the debug
package has no dependencies from the rest of ``skilleval/``.
"""

from skilleval.debug.wrapper import DebugModel

__all__ = ["DebugModel"]
