import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qwen36_38 as runtime


class QwenEventLedgerTests(unittest.TestCase):
    def request(self):
        return {
            "director_backend": "qwen3.8",
            "chunk_number": 2,
            "chunk_count": 3,
            "original_prompt": "prompt",
            "target_shots": [],
            "mandatory_coverage": [{"id": "S1.V2", "action": "open the door"}],
            "previous_event_ledger": {
                "completed": [{"id": "S1.V1", "summary": "stood up"}],
                "active": [], "pending": [],
                "forbidden": [{"id": "S1.V1", "summary": "stood up"}],
            },
        }

    def test_chunk_prompt_includes_previous_event_ownership(self):
        _system, prompt = runtime._chunk_messages(self.request())
        self.assertIn("EVENT OWNERSHIP LEDGER", prompt)
        self.assertIn("S1.V1", prompt)
        self.assertIn("MUST NOT stage, replay", prompt)

    def test_parser_normalizes_and_forbids_completed_events(self):
        result = runtime._chunk_prompt({
            "confidence": "high", "analysis": "ok", "detailed_description": "continue",
            "event_ledger": {
                "completed": [{"id": "S1.V1", "summary": "stood up"}],
                "active": [{"id": "S1.V2", "summary": "walking"}],
                "pending": [{"id": "S1.V3", "summary": "open door"}],
                "forbidden": [],
            },
        }, "{}", "system", "prompt", self.request())
        self.assertEqual(result.event_ledger["completed"][0]["id"], "S1.V1")
        self.assertEqual(result.event_ledger["forbidden"][0]["id"], "S1.V1")

    def test_parser_rejects_conflicting_event_ownership(self):
        with self.assertRaisesRegex(runtime.Qwen35ObservationError, "appears in both"):
            runtime._event_ledger({
                "completed": [],
                "active": [{"id": "S1.V1", "summary": "walking"}],
                "pending": [{"id": "S1.V1", "summary": "walking later"}],
                "forbidden": [],
            })

    def test_payload_round_trip_preserves_ledger(self):
        original = runtime._chunk_prompt({
            "detailed_description": "continue",
            "event_ledger": {"completed": [], "active": [], "pending": [], "forbidden": []},
        }, "{}", "system", "prompt")
        restored = runtime._from_payload(runtime._payload(original), False)
        self.assertEqual(restored.event_ledger, original.event_ledger)


if __name__ == "__main__":
    unittest.main()
