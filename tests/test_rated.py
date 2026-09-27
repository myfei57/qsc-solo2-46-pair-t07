"""Unit and integration tests for the rated-capacity guard."""

from __future__ import annotations

import unittest
from pathlib import Path

from cranefleet.audit import Audit
from cranefleet.console import Console, Request
from cranefleet.errors import ChartError, LoadOverrated
from cranefleet.rated import (
    LEVEL_OK,
    LEVEL_OVERLOAD,
    LEVEL_WARN,
    ChartPoint,
    LoadChart,
    LoadMonitor,
    acknowledge,
    default_chart,
    evaluate,
    guard,
    rated_load,
    update_reading,
)
from cranefleet.wiring import build_app

WEB_ROOT = Path(__file__).resolve().parents[1] / "web"


def small_chart() -> LoadChart:
    return LoadChart(points=(ChartPoint(10.0, 8.0), ChartPoint(20.0, 4.0)))


def dispatch(console: Console, method: str, path: str, body: dict | None = None):  # noqa: ANN201
    request = Request(method=method, path=path, query={}, body=body or {}, body_valid=True)
    return console.dispatch(request)


class ChartTests(unittest.TestCase):
    def test_chart_validation(self) -> None:
        with self.assertRaises(ChartError):
            LoadChart(points=(ChartPoint(10.0, 8.0),))
        with self.assertRaises(ChartError):
            LoadChart(points=(ChartPoint(20.0, 4.0), ChartPoint(10.0, 8.0)))
        with self.assertRaises(ChartError):
            LoadChart(points=(ChartPoint(10.0, 4.0), ChartPoint(20.0, 8.0)))
        with self.assertRaises(ChartError):
            LoadChart(points=(ChartPoint(10.0, 0.0), ChartPoint(20.0, 4.0)))



    def test_rated_load_interpolates_between_points(self) -> None:
        chart = small_chart()
        self.assertEqual(rated_load(chart, 10.0), 8.0)
        self.assertEqual(rated_load(chart, 20.0), 4.0)
        self.assertAlmostEqual(rated_load(chart, 15.0), 6.0)
        self.assertAlmostEqual(rated_load(chart, 12.5), 7.0)

    def test_rated_load_never_extrapolates(self) -> None:
        chart = small_chart()
        with self.assertRaises(ChartError):
            rated_load(chart, 9.9)
        with self.assertRaises(ChartError):
            rated_load(chart, 20.1)

    def test_default_chart_falls_with_radius(self) -> None:
        chart = default_chart()
        capacities = [point.capacity for point in chart.points]
        self.assertEqual(capacities, sorted(capacities, reverse=True))
        self.assertEqual(rated_load(chart, 30.0), 2.7)


class EvaluateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chart = small_chart()

    def test_ok_level_reports_margin(self) -> None:
        decision = evaluate(self.chart, 10.0, 4.0)
        self.assertEqual(decision.level, LEVEL_OK)
        self.assertEqual(decision.capacity, 8.0)
        self.assertEqual(decision.margin, 4.0)
        self.assertEqual(decision.ratio, 0.5)
        self.assertIn("余量", decision.message)
        self.assertIn("level=ok", decision.summary())

    def test_warn_level_near_capacity(self) -> None:
        decision = evaluate(self.chart, 10.0, 7.5)
        self.assertEqual(decision.level, LEVEL_WARN)
        self.assertAlmostEqual(decision.margin, 0.5)
        self.assertIn("余量仅剩", decision.message)

    def test_exact_capacity_is_still_a_warning(self) -> None:
        decision = evaluate(self.chart, 10.0, 8.0)
        self.assertEqual(decision.level, LEVEL_WARN)
        self.assertEqual(decision.margin, 0.0)

    def test_overload_level_forbids_hoisting(self) -> None:
        decision = evaluate(self.chart, 10.0, 8.4)
        self.assertEqual(decision.level, LEVEL_OVERLOAD)
        self.assertAlmostEqual(decision.margin, -0.4)
        self.assertIn("禁止起升", decision.message)


class GuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.monitor = LoadMonitor("crane-1", chart=small_chart())
        self.audit = Audit(50)

    def test_update_reading_stores_sample_and_decision(self) -> None:
        decision = update_reading(self.monitor, 15.0, 3.0)
        self.assertIsNotNone(decision)
        self.assertEqual(self.monitor.radius_now(), 15.0)
        self.assertEqual(self.monitor.load_now(), 3.0)
        self.assertEqual(self.monitor.reading_seq_now(), 1)
        self.assertEqual(self.monitor.decision_now(), decision)
        self.assertEqual(self.monitor.fault_now(), "")

    def test_out_of_span_reading_latches_a_fault(self) -> None:
        self.assertIsNone(update_reading(self.monitor, 25.0, 1.0))
        self.assertIn("outside chart span", self.monitor.fault_now())
        self.assertIsNone(self.monitor.decision_now())
        with self.assertRaises(LoadOverrated):
            guard(self.monitor, self.audit)

    def test_guard_blocks_without_a_reading(self) -> None:
        with self.assertRaises(LoadOverrated):
            guard(self.monitor)

    def test_guard_passes_ok_and_records(self) -> None:
        update_reading(self.monitor, 15.0, 3.0)
        decision = guard(self.monitor, self.audit)
        self.assertEqual(decision.level, LEVEL_OK)
        entry = self.audit.latest("rated-guard")
        self.assertIsNotNone(entry)
        self.assertIn("pass", entry.detail)

    def test_warn_holds_until_acknowledged(self) -> None:
        update_reading(self.monitor, 10.0, 7.5)
        with self.assertRaises(LoadOverrated):
            guard(self.monitor, self.audit)
        self.assertTrue(acknowledge(self.monitor))
        self.assertTrue(self.monitor.acknowledged_now())
        decision = guard(self.monitor, self.audit)
        self.assertEqual(decision.level, LEVEL_WARN)
        details = [entry.detail for entry in self.audit.filter_by_actions(["rated-guard"])]
        self.assertTrue(any("hold" in detail for detail in details))
        self.assertTrue(any("pass" in detail for detail in details))

    def test_new_reading_invalidates_the_acknowledgement(self) -> None:
        update_reading(self.monitor, 10.0, 7.5)
        acknowledge(self.monitor)
        update_reading(self.monitor, 10.0, 7.6)
        self.assertFalse(self.monitor.acknowledged_now())
        with self.assertRaises(LoadOverrated):
            guard(self.monitor)

    def test_overload_cannot_be_acknowledged(self) -> None:
        update_reading(self.monitor, 10.0, 9.0)
        self.assertFalse(acknowledge(self.monitor))
        with self.assertRaises(LoadOverrated):
            guard(self.monitor, self.audit)
        entry = self.audit.latest("rated-guard")
        self.assertIn("block", entry.detail)


class RatedApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = build_app("memory", web_root=str(WEB_ROOT))
        self.console = self.app.console

    def tearDown(self) -> None:
        self.app.shutdown()

    def _raise(self):  # noqa: ANN202
        return dispatch(
            self.console,
            "POST",
            "/api/crane/command",
            {"source": "cab", "action": "raise", "axis": "hoist"},
        )

    def test_update_then_state_reports_decision(self) -> None:
        status, payload = dispatch(
            self.console, "POST", "/api/rated/update", {"radius": 30, "load": 2.0}
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["data"]["level"], "ok")
        self.assertEqual(payload["data"]["capacity"], 2.7)
        self.assertAlmostEqual(payload["data"]["margin"], 0.7)
        status, payload = dispatch(self.console, "GET", "/api/rated/state")
        self.assertEqual(status, 200)
        self.assertEqual(payload["data"]["radius"], 30.0)
        self.assertEqual(payload["data"]["decision"]["level"], "ok")
        self.assertEqual(len(payload["data"]["chart"]["points"]), 9)
        request = Request("GET", "/api/audit", {"action": ["rated-check"]}, {})
        _, payload = self.console.dispatch(request)
        self.assertEqual(len(payload["data"]), 1)
        self.assertIn("margin=0.70t", payload["data"][0]["detail"])

    def test_update_rejects_bad_input(self) -> None:
        status, _ = dispatch(self.console, "POST", "/api/rated/update", {"radius": -1, "load": 2})
        self.assertEqual(status, 400)
        request = Request("POST", "/api/rated/update", {}, {}, body_valid=False)
        status, _ = self.console.dispatch(request)
        self.assertEqual(status, 400)

    def test_out_of_span_update_latches_fault_and_blocks_raise(self) -> None:
        status, payload = dispatch(
            self.console, "POST", "/api/rated/update", {"radius": 80, "load": 1.0}
        )
        self.assertEqual(status, 409)
        self.assertIn("outside chart span", payload["error"])
        status, payload = self._raise()
        self.assertEqual(status, 409)
        self.assertIn("fault", payload["error"])
        _, payload = dispatch(self.console, "GET", "/api/rated/state")
        self.assertIn("outside chart span", payload["data"]["fault"])

    def test_raise_blocked_without_a_reading(self) -> None:
        status, payload = self._raise()
        self.assertEqual(status, 409)
        self.assertEqual(payload["error"], "no load reading")

    def test_raise_passes_when_within_capacity(self) -> None:
        dispatch(self.console, "POST", "/api/rated/update", {"radius": 30, "load": 2.0})
        status, payload = self._raise()
        self.assertEqual(status, 200)
        self.assertEqual(payload["data"]["action"], "raise")
        request = Request("GET", "/api/audit", {"action": ["rated-guard"]}, {})
        _, payload = self.console.dispatch(request)
        self.assertTrue(any("pass" in entry["detail"] for entry in payload["data"]))

    def test_warn_holds_raise_until_acknowledged(self) -> None:
        dispatch(self.console, "POST", "/api/rated/update", {"radius": 30, "load": 2.5})
        status, payload = self._raise()
        self.assertEqual(status, 409)
        self.assertIn("余量仅剩 0.20 t", payload["error"])
        status, payload = dispatch(self.console, "POST", "/api/rated/ack")
        self.assertEqual(status, 200)
        self.assertTrue(payload["data"]["acknowledged"])
        status, _ = self._raise()
        self.assertEqual(status, 200)

    def test_overload_cuts_motion_and_forbids_raise(self) -> None:
        dispatch(
            self.console,
            "POST",
            "/api/crane/command",
            {"source": "cab", "action": "lower", "axis": "hoist"},
        )
        _, payload = dispatch(self.console, "GET", "/api/hoist/state")
        self.assertTrue(payload["data"]["moving"])
        status, payload = dispatch(
            self.console, "POST", "/api/rated/update", {"radius": 48, "load": 2.0}
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["data"]["level"], "overload")
        _, payload = dispatch(self.console, "GET", "/api/hoist/state")
        self.assertFalse(payload["data"]["moving"])
        status, payload = dispatch(self.console, "POST", "/api/rated/ack")
        self.assertEqual(status, 409)
        status, payload = self._raise()
        self.assertEqual(status, 409)
        self.assertIn("禁止起升", payload["error"])
        request = Request("GET", "/api/audit", {"action": ["rated-cut"]}, {})
        _, payload = self.console.dispatch(request)
        self.assertEqual(len(payload["data"]), 1)

    def test_check_endpoint_mirrors_the_gate(self) -> None:
        status, payload = dispatch(self.console, "POST", "/api/rated/check")
        self.assertEqual(status, 409)
        dispatch(self.console, "POST", "/api/rated/update", {"radius": 30, "load": 2.0})
        status, payload = dispatch(self.console, "POST", "/api/rated/check")
        self.assertEqual(status, 200)
        self.assertEqual(payload["data"]["level"], "ok")

    def test_state_snapshot_contains_rated(self) -> None:
        status, payload = dispatch(self.console, "GET", "/api/state")
        self.assertEqual(status, 200)
        self.assertIn("rated", payload["data"])
        self.assertEqual(payload["data"]["rated"]["warn_ratio"], 0.9)


if __name__ == "__main__":
    unittest.main()
