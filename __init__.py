"""SingSync ComfyUI custom node pack.

ComfyUI imports this file when the folder is dropped into ``custom_nodes``. A
broken optional dependency must not stop ComfyUI from starting, so the node
registration is guarded and only reports the problem.
"""
from __future__ import annotations

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}

try:
    from .singsync.nodes import (NODE_CLASS_MAPPINGS as _CLASSES,
                                 NODE_DISPLAY_NAME_MAPPINGS as _NAMES)
    NODE_CLASS_MAPPINGS.update(_CLASSES)
    NODE_DISPLAY_NAME_MAPPINGS.update(_NAMES)
except Exception as exc:  # pragma: no cover - keep ComfyUI booting
    print("[SingSync] nodes not registered: %s: %s" % (type(exc).__name__, exc))

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
