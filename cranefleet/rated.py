"""Rated-capacity guard: load chart lookup, margin judgement and hoist gate.

Every lift is judged against the crane's load chart (起重特性表): the current
radius selects the rated capacity, a load above ``WARN_RATIO`` of that
capacity is held back until the operator acknowledges the remaining margin,
and a load above the capacity itself forbids hoisting outright. Each
judgement is funnelled through the audit log so the console keeps a record of
every decision.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from .audit import Audit
from .errors import ChartError, LoadOverrated

WARN_RATIO = 0.9

LEVEL_OK = "ok"
LEVEL_WARN = "warn"
LEVEL_OVERLOAD = "overload"


@dataclass(frozen=True)
class ChartPoint:
    """One (radius, rated capacity) pair of the load chart."""

    radius: float
    capacity: float

    def to_dict(self) -> dict[str, float]:
        return {"radius": self.radius, "capacity": self.capacity}


@dataclass(frozen=True)
class LoadChart:
    """The rated load chart of one crane, capacity falling with radius."""

    points: tuple[ChartPoint, ...]

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ChartError("load chart needs at least two points")
        radii = [point.radius for point in self.points]
        capacities = [point.capacity for point in self.points]
        if any(radius <= 0 for radius in radii):
            raise ChartError("chart radii must be positive")
        if any(b <= a for a, b in zip(radii, radii[1:])):
            raise ChartError("chart radii must be strictly increasing")
        if any(capacity <= 0 for capacity in capacities):
            raise ChartError("chart capacities must be positive")
        if any(b > a for a, b in zip(capacities, capacities[1:])):
            raise ChartError("chart capacities must not increase with radius")

    def to_dict(self) -> dict[str, object]:
        return {"points": [point.to_dict() for point in self.points]}


def default_chart() -> LoadChart:
    """The factory chart of the site crane (QTZ80 class, tonnes vs metres)."""

    return LoadChart(
        points=(
            ChartPoint(10.0, 8.0),
            ChartPoint(14.0, 6.0),
            ChartPoint(18.0, 4.8),
            ChartPoint(24.0, 3.5),
            ChartPoint(30.0, 2.7),
            ChartPoint(36.0, 2.1),
            ChartPoint(42.0, 1.7),
            ChartPoint(48.0, 1.4),
            ChartPoint(54.0, 1.1),
        )
    )


def rated_load(chart: LoadChart, radius: float) -> float:
    """Rated capacity at ``radius``, interpolated between chart points.

    A radius outside the chart span is never extrapolated: guessing a
    capacity there is exactly how trolleys get tipped over.
    """

    points = chart.points
    if radius < points[0].radius or radius > points[-1].radius:
        raise ChartError(
            f"radius {radius:.2f} m outside chart span "
            f"{points[0].radius:.1f}-{points[-1].radius:.1f} m"
        )
    for previous, following in zip(points, points[1:]):
        if previous.radius <= radius <= following.radius:
            fraction = (radius - previous.radius) / (following.radius - previous.radius)
            return previous.capacity + (following.capacity - previous.capacity) * fraction
    return points[-1].capacity  # unreachable: the span check bounds the loop


@dataclass(frozen=True)
class LoadDecision:
    """The outcome of judging one load against the chart."""

    level: str
    radius: float
    load: float
    capacity: float
    ratio: float
    margin: float
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "level": self.level,
            "radius": self.radius,
            "load": self.load,
            "capacity": self.capacity,
            "ratio": self.ratio,
            "margin": self.margin,
            "message": self.message,
        }

    def summary(self) -> str:
        return (
            f"level={self.level} radius={self.radius:.2f}m load={self.load:.2f}t "
            f"capacity={self.capacity:.2f}t margin={self.margin:.2f}t"
        )


def evaluate(
    chart: LoadChart, radius: float, load: float, warn_ratio: float = WARN_RATIO
) -> LoadDecision:
    """Judge ``load`` at ``radius`` against the chart."""

    capacity = rated_load(chart, radius)
    ratio = load / capacity
    margin = capacity - load
    if ratio > 1.0:
        level = LEVEL_OVERLOAD
        message = (
            f"超载禁止起升：幅度 {radius:.1f} m 允许 {capacity:.2f} t，"
            f"当前 {load:.2f} t，超出 {-margin:.2f} t"
        )
    elif ratio >= warn_ratio:
        level = LEVEL_WARN
        message = (
            f"接近额定：幅度 {radius:.1f} m 允许 {capacity:.2f} t，"
            f"余量仅剩 {margin:.2f} t，确认后方可起升"
        )
    else:
        level = LEVEL_OK
        message = f"正常：幅度 {radius:.1f} m 允许 {capacity:.2f} t，余量 {margin:.2f} t"
    return LoadDecision(
        level=level,
        radius=radius,
        load=load,
        capacity=capacity,
        ratio=ratio,
        margin=margin,
        message=message,
    )


class LoadMonitor:
    """Mutable state of the rated-capacity guard of one crane."""

    def __init__(
        self,
        crane_id: str,
        chart: LoadChart | None = None,
        warn_ratio: float = WARN_RATIO,
    ) -> None:
        self._lock = threading.Lock()
        self.id = crane_id
        self.chart = chart if chart is not None else default_chart()
        self.warn_ratio = warn_ratio
        self.radius = 0.0
        self.load = 0.0
        self.reading_seq = 0
        self.last_decision: LoadDecision | None = None
        self.fault = ""
        self.ack_seq = 0

    def radius_now(self) -> float:
        with self._lock:
            return self.radius

    def load_now(self) -> float:
        with self._lock:
            return self.load

    def reading_seq_now(self) -> int:
        with self._lock:
            return self.reading_seq

    def decision_now(self) -> LoadDecision | None:
        with self._lock:
            return self.last_decision

    def fault_now(self) -> str:
        with self._lock:
            return self.fault

    def acknowledged_now(self) -> bool:
        with self._lock:
            return self.reading_seq > 0 and self.ack_seq >= self.reading_seq


def update_reading(monitor: LoadMonitor, radius: float, load: float) -> LoadDecision | None:
    """Store the latest sensor sample and judge it immediately.

    Returns the decision, or ``None`` when the radius falls outside the
    chart; the monitor then latches a fault so the hoist gate stays shut.
    """

    try:
        decision = evaluate(monitor.chart, radius, load, monitor.warn_ratio)
    except ChartError as exc:
        with monitor._lock:  # noqa: SLF001 - module-level functions share monitor state
            monitor.radius = radius
            monitor.load = load
            monitor.reading_seq += 1
            monitor.last_decision = None
            monitor.fault = str(exc)
        return None
    with monitor._lock:  # noqa: SLF001
        monitor.radius = radius
        monitor.load = load
        monitor.reading_seq += 1
        monitor.last_decision = decision
        monitor.fault = ""
    return decision


def acknowledge(monitor: LoadMonitor) -> bool:
    """Confirm the current warning; only valid for the current reading."""

    with monitor._lock:  # noqa: SLF001
        if monitor.last_decision is None or monitor.last_decision.level != LEVEL_WARN:
            return False
        monitor.ack_seq = monitor.reading_seq
        return True


def guard(monitor: LoadMonitor, recorder: Audit | None = None) -> LoadDecision:
    """Gate a hoisting command on the latest judgement.

    Every call leaves an audit entry. Returns the decision when hoisting may
    proceed; raises :class:`LoadOverrated` when it must not.
    """

    with monitor._lock:  # noqa: SLF001
        fault = monitor.fault
        decision = monitor.last_decision
        acknowledged = monitor.ack_seq >= monitor.reading_seq
        crane_id = monitor.id
    if fault:
        _record(recorder, crane_id, "block fault=" + fault)
        raise LoadOverrated("rated guard fault: " + fault)
    if decision is None:
        _record(recorder, crane_id, "block no-reading")
        raise LoadOverrated("no load reading")
    if decision.level == LEVEL_OVERLOAD:
        _record(recorder, crane_id, "block " + decision.summary())
        raise LoadOverrated(decision.message)
    if decision.level == LEVEL_WARN and not acknowledged:
        _record(recorder, crane_id, "hold " + decision.summary())
        raise LoadOverrated(decision.message)
    _record(recorder, crane_id, "pass " + decision.summary())
    return decision


def _record(recorder: Audit | None, crane_id: str, detail: str) -> None:
    if recorder is not None:
        recorder.record(crane_id, "rated-guard", detail)
