"""Unit and API tests for the rated-capacity limiter."""

from __future__ import annotations

import unittest
from pathlib import Path

from cranefleet.audit import Audit
from cranefleet.console import Request
from cranefleet.errors import OverRated
from cranefleet.rating import (
    DEFAULT_CHART,
    STATUS_CAUTION,
    STATUS_OK,
    STATUS_OVER,
    WARN_RATIO,
    Rating,
    capacity_at,
    check,
    guard_raise,
)
from cranefleet.wiring import build_app

WEB_ROOT = Path(__file__).resolve().parents[1] / "web"


def dispatch(console, method: str, path: str, body: dict | None = None):  # noqa: ANN001, ANN201
    request = Request(method=method, path=path, query={}, body=body or {}, body_valid=True)
    return console.dispatch(request)


class CapacityChartTests(unittest.TestCase):
    def test_breakpoints_and_interpolation(self) -> None:
        self.assertEqual(capacity_at(DEFAULT_CHART, 10.0), 8.0)
        self.assertEqual(capacity_at(DEFAULT_CHART, 50.0), 1.2)
        self.assertAlmostEqual(capacity_at(DEFAULT_CHART, 13.0), 7.0)
        self.assertAlmostEqual(capacity_at(DEFAULT_CHART, 28.0), 3.4)

    def test_out_of_chart_radius(self) -> None:
        self.assertEqual(capacity_at(DEFAULT_CHART, 5.0), 8.0)
        self.assertEqual(capacity_at(DEFAULT_CHART, 50.1), 0.0)
        self.assertEqual(capacity_at((), 10.0), 0.0)


class RatingCheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rating = Rating("crane-1")
        self.audit = Audit(50)

    def test_ok_caution_and_over(self) -> None:
        verdict = check(self.rating, 24.0, 2.0, self.audit)
        self.assertEqual(verdict.status, STATUS_OK)
        self.assertEqual(verdict.capacity, 4.0)
        self.assertAlmostEqual(verdict.margin, 2.0)

        verdict = check(self.rating, 24.0, 4.0 * WARN_RATIO, self.audit)
        self.assertEqual(verdict.status, STATUS_CAUTION)
        self.assertAlmostEqual(verdict.margin, 0.4)

        verdict = check(self.rating, 24.0, 4.0, self.audit)
        self.assertEqual(verdict.status, STATUS_CAUTION)
        self.assertAlmostEqual(verdict.margin, 0.0)

        verdict = check(self.rating, 24.0, 4.4, self.audit)
        self.assertEqual(verdict.status, STATUS_OVER)
        self.assertAlmostEqual(verdict.margin, -0.4)

    def test_beyond_chart_radius_is_over(self) -> None:
        verdict = check(self.rating, 55.0, 0.5, self.audit)
        self.assertEqual(verdict.status, STATUS_OVER)
        self.assertEqual(verdict.capacity, 0.0)

    def test_every_check_is_recorded(self) -> None:
        check(self.rating, 16.0, 3.0, self.audit)
        check(self.rating, 16.0, 5.5, self.audit)
        entries = self.audit.filter_by_actions(["rating-check"])
        self.assertEqual(len(entries), 2)
        self.assertIn("margin=0.50t", entries[0].detail)
        self.assertIn("status=caution", entries[0].detail)
        self.assertEqual(self.rating.sequence_now(), 2)
        self.assertEqual(self.rating.status_now(), STATUS_CAUTION)

    def test_guard_raise_blocks_only_over(self) -> None:
        check(self.rating, 24.0, 3.8, self.audit)
        verdict = guard_raise(self.rating, self.audit)
        self.assertEqual(verdict.status, STATUS_CAUTION)

        check(self.rating, 24.0, 4.4, self.audit)
        with self.assertRaises(OverRated):
            guard_raise(self.rating, self.audit)
        actions = [entry.action for entry in self.audit.snapshot()]
        self.assertIn("rating-blocked", actions)
        blocked = self.audit.latest("rating-blocked")
        self.assertIn("4.40t", blocked.detail)
        self.assertIn("4.00t", blocked.detail)


class RatingApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = build_app("memory", web_root=str(WEB_ROOT))
        self.console = self.app.console

    def tearDown(self) -> None:
        self.app.shutdown()

    def test_check_endpoint_updates_state(self) -> None:
        status, payload = dispatch(
            self.console, "POST", "/api/rating/check", {"radius": 24, "load": 3.6}
        )
        self.assertEqual(status, 200)
        verdict = payload["data"]["verdict"]
        self.assertEqual(verdict["status"], "caution")
        self.assertAlmostEqual(verdict["capacity"], 4.0)
        self.assertAlmostEqual(verdict["margin"], 0.4)

        status, payload = dispatch(self.console, "GET", "/api/rating/state")
        self.assertEqual(status, 200)
        self.assertEqual(payload["data"]["radius"], 24.0)
        self.assertEqual(payload["data"]["status"], "caution")
        self.assertEqual(len(payload["data"]["chart"]), len(DEFAULT_CHART))

        status, payload = dispatch(self.console, "GET", "/api/state")
        self.assertIn("rating", payload["data"])
        self.assertEqual(payload["data"]["rating"]["load"], 3.6)

    def test_check_endpoint_validates_body(self) -> None:
        status, _ = dispatch(self.console, "POST", "/api/rating/check", {"radius": 0, "load": 1})
        self.assertEqual(status, 400)
        status, _ = dispatch(self.console, "POST", "/api/rating/check", {"radius": 10, "load": -1})
        self.assertEqual(status, 400)
        request = Request("POST", "/api/rating/check", {}, {}, body_valid=False)
        status, _ = self.console.dispatch(request)
        self.assertEqual(status, 400)

    def test_raise_is_blocked_when_over(self) -> None:
        dispatch(self.console, "POST", "/api/rating/check", {"radius": 24, "load": 4.4})
        status, payload = dispatch(
            self.console, "POST", "/api/crane/command", {"source": "cab", "action": "raise"}
        )
        self.assertEqual(status, 409)
        self.assertIn("over rated capacity", payload["error"])
        self.assertFalse(self.console.hoist.is_moving())
        actions = [entry.action for entry in self.console.audit.snapshot()]
        self.assertIn("rating-check", actions)
        self.assertIn("rating-blocked", actions)

    def test_raise_passes_with_margin_and_caution_is_reported(self) -> None:
        dispatch(self.console, "POST", "/api/rating/check", {"radius": 24, "load": 3.8})
        status, payload = dispatch(
            self.console, "POST", "/api/crane/command", {"source": "cab", "action": "raise"}
        )
        self.assertEqual(status, 200)
        self.assertTrue(self.console.hoist.is_moving())
        self.assertEqual(payload["data"]["rating"]["status"], "caution")
        self.assertAlmostEqual(payload["data"]["rating"]["margin"], 0.2)

    def test_raise_without_check_uses_empty_hook(self) -> None:
        status, payload = dispatch(
            self.console, "POST", "/api/crane/command", {"source": "cab", "action": "raise"}
        )
        self.assertEqual(status, 200)
        self.assertNotIn("rating", payload["data"])


if __name__ == "__main__":
    unittest.main()
