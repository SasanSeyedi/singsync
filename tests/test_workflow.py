"""The shipped workflow JSON must match the nodes it references."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from singsync import nodes

WORKFLOW = Path(__file__).resolve().parents[1] / "workflows" / "singsync_music_video.json"


def widget_names(cls) -> list[str]:
    """Inputs ComfyUI serialises into ``widgets_values``, in order."""
    spec = cls.INPUT_TYPES()
    names = []
    for section in ("required", "optional"):
        for name, definition in (spec.get(section) or {}).items():
            kind = definition[0]
            if isinstance(kind, list) or kind in ("STRING", "INT", "FLOAT", "BOOLEAN"):
                names.append(name)
    return names


class WorkflowJson(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))

    def test_node_ids_are_unique(self):
        ids = [n["id"] for n in self.workflow["nodes"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(self.workflow["last_node_id"], max(ids))

    def test_every_node_type_is_registered(self):
        for node in self.workflow["nodes"]:
            self.assertIn(node["type"], nodes.NODE_CLASS_MAPPINGS, node["type"])

    def test_widget_values_match_the_node_inputs(self):
        for node in self.workflow["nodes"]:
            cls = nodes.NODE_CLASS_MAPPINGS[node["type"]]
            self.assertEqual(len(node.get("widgets_values", [])),
                             len(widget_names(cls)),
                             "%s: %s" % (node["type"], widget_names(cls)))

    def test_outputs_match_the_node_returns(self):
        for node in self.workflow["nodes"]:
            cls = nodes.NODE_CLASS_MAPPINGS[node["type"]]
            self.assertEqual([o["name"] for o in node["outputs"]],
                             list(cls.RETURN_NAMES), node["type"])
            self.assertEqual([o["type"] for o in node["outputs"]],
                             list(cls.RETURN_TYPES), node["type"])

    def test_inputs_exist_on_the_node(self):
        try:
            import torch  # noqa: F401
            have_torch = True
        except ImportError:
            have_torch = False
        for node in self.workflow["nodes"]:
            spec = nodes.NODE_CLASS_MAPPINGS[node["type"]].INPUT_TYPES()
            known = {**(spec.get("required") or {}), **(spec.get("optional") or {})}
            for slot in node.get("inputs", []):
                if slot["name"] not in known and slot["type"] == "IMAGE" \
                        and not have_torch:
                    # IMAGE sockets are only offered where torch is installed,
                    # which is always true inside ComfyUI but not in CI.
                    continue
                self.assertIn(slot["name"], known, node["type"])
                self.assertEqual(known[slot["name"]][0], slot["type"], slot["name"])

    def test_links_are_consistent(self):
        by_id = {n["id"]: n for n in self.workflow["nodes"]}
        seen = set()
        for link_id, src_id, src_slot, dst_id, dst_slot, link_type in \
                self.workflow["links"]:
            self.assertNotIn(link_id, seen)
            seen.add(link_id)
            source = by_id[src_id]["outputs"][src_slot]
            target = by_id[dst_id]["inputs"][dst_slot]
            self.assertEqual(source["type"], link_type)
            self.assertEqual(target["type"], link_type)
            self.assertEqual(target["link"], link_id)
            self.assertIn(link_id, source["links"] or [])
        self.assertEqual(self.workflow["last_link_id"], max(seen))

    def test_the_chain_runs_analyze_to_assemble(self):
        types = [n["type"] for n in sorted(self.workflow["nodes"],
                                           key=lambda n: n["order"])]
        self.assertEqual(types[0], "SingSyncSongAnalyze")
        self.assertEqual(types[-1], "SingSyncAssemble")
        self.assertEqual(len(self.workflow["links"]), len(self.workflow["nodes"]) - 1)


if __name__ == "__main__":
    unittest.main()
