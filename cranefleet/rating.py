"""Rated capacity by radius: chart lookup, caution margin and the hoist gate.

The limiter answers one question for the operator: at the current radius
(幅度), how much may the crane lift (额定起重量)? Every determination is
written to the audit log so a near miss can be reconstructed afterwards.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from .audit import Audit
from .errors import OverRated

WARN_RATIO = 0.9

# (radius m, capacity t) breakpoints of the load chart, ascending by radius.
DEFAULT_CHART = (
    (10.0, 8.0),
    (16.0, 6.0),
    (24.0, 4.0),
    (32.0, 2.8),
    (40.0, 2.0),
    (50.0, 1.2),
)

STATUS_OK = "ok"
STATUS_CAUTION = "caution"
STATUS_OVER = "over"


class Rating:
    """Mutable state of the rated-capacity limiter."""

    def __init__(
        self,
        rating_id: str,
        chart: tuple[tuple[float, float], ...] = DEFAULT_CHART,
        warn_ratio: float = WARN_RATIO,
    ) -> None:
        self._lock = threading.Lock()
        self.id = rating_id
        self.chart = tuple(sorted(chart))
        self.warn_ratio = warn_ratio
        self.radius = 0.0
        self.load = 0.0
        self.sequence = 0
        self.status = STATUS_OK

    def radius_now(self) -> float:
        with self._lock:
            return self.radius

    def load_now(self) -> float:
        with self._lock:
            return self.load

    def sequence_now(self) -> int:
        with self._lock:
            return self.sequence

    def status_now(self) -> str:
        with self._lock:
            return self.status

    def chart_points(self) -> list[tuple[float, float]]:
        with self._lock:
            return list(self.chart)


@dataclass(frozen=True)
class Verdict:
    """The outcome of one rated-capacity determination."""

    radius: float
    load: float
    capacity: float
    margin: float
    ratio: float
    status: str
    sequence: int

    def to_dict(self) -> dict[str, object]:
        return {
            "radius": self.radius,
            "load": self.load,
            "capacity": self.capacity,
            "margin": round(self.margin, 3),
            "ratio": round(self.ratio, 3),
            "status": self.status,
            "sequence": self.sequence,
        }


def capacity_at(chart: tuple[tuple[float, float], ...], radius: float) -> float:
    """Interpolate the rated capacity for ``radius`` from the load chart.

    Below the first breakpoint the first capacity applies; past the last
    breakpoint the radius is out of chart and nothing may be lifted.
    """

    points = tuple(sorted(chart))
    if not points:
        return 0.0
    if radius <= points[0][0]:
        return points[0][1]
    if radius > points[-1][0]:
        return 0.0
    for (inner_radius, inner_capacity), (outer_radius, outer_capacity) in zip(points, points[1:]):
        if radius <= outer_radius:
            span = outer_radius - inner_radius
            if span <= 0:
                return outer_capacity
            fraction = (radius - inner_radius) / span
            return inner_capacity + fraction * (outer_capacity - inner_capacity)
    return points[-1][1]


def _status_for(ratio: float, warn_ratio: float) -> str:
    if ratio > 1.0:
        return STATUS_OVER
    if ratio >= warn_ratio:
        return STATUS_CAUTION
    return STATUS_OK


def check(rating: Rating, radius: float, load: float, recorder: Audit | None = None) -> Verdict:
    """Determine the verdict for ``load`` at ``radius`` and record it."""

    with rating._lock:  # noqa: SLF001 - module-level functions share rating state
        rating.radius = radius
        rating.load = load
        rating.sequence += 1
        sequence = rating.sequence
        capacity = capacity_at(rating.chart, radius)
        ratio = load / capacity if capacity > 0 else float("inf")
        status = _status_for(ratio, rating.warn_ratio)
        rating.status = status
        margin = capacity - load
        warn_ratio = rating.warn_ratio
    verdict = Verdict(
        radius=radius,
        load=load,
        capacity=capacity,
        margin=margin,
        ratio=ratio,
        status=status,
        sequence=sequence,
    )
    if recorder is not None:
        recorder.record(
            rating.id,
            "rating-check",
            f"radius={radius:.1f}m load={load:.2f}t capacity={capacity:.2f}t "
            f"margin={margin:.2f}t warn_at={warn_ratio:.2f} status={status}",
        )
    return verdict


def guard_raise(rating: Rating, recorder: Audit | None = None) -> Verdict:
    """Re-check the stored radius/load before a raise; refuse when over."""

    with rating._lock:  # noqa: SLF001
        radius = rating.radius
        load = rating.load
    verdict = check(rating, radius, load, recorder)
    if verdict.status == STATUS_OVER:
        if recorder is not None:
            recorder.record(
                rating.id,
                "rating-blocked",
                f"raise blocked: load={verdict.load:.2f}t exceeds capacity="
                f"{verdict.capacity:.2f}t at radius={verdict.radius:.1f}m",
            )
        raise OverRated(
            f"over rated capacity: load {verdict.load:.2f}t > {verdict.capacity:.2f}t "
            f"at radius {verdict.radius:.1f}m"
        )
    return verdict
