import unittest

from mesh.orchestrator.wayfinder_tools import _emit_progress


class PlanningProgressTests(unittest.IsolatedAsyncioTestCase):
    async def test_emits_customer_safe_stage_event(self):
        received = []

        async def callback(event):
            received.append(event)

        await _emit_progress(callback, "DISCOVERY", "IN_PROGRESS", "Exploring experiences.")

        self.assertEqual(received, [{
            "type": "stage_update",
            "stage": "DISCOVERY",
            "status": "IN_PROGRESS",
            "message": "Exploring experiences.",
        }])

    async def test_progress_callback_failure_does_not_break_planning(self):
        async def disconnected_callback(_event):
            raise RuntimeError("client disconnected")

        await _emit_progress(
            disconnected_callback, "TRANSPORT", "IN_PROGRESS", "Searching flights."
        )


if __name__ == "__main__":
    unittest.main()
