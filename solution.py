## Student Name: Nathan Binu Edappilly
## Student ID: 219317965

"""
Task A: Appointment Timeslot Recommender (Stub)

In this lab, you will design and implement an Appointment Slot Recommender using an LLM assistant
as your primary programming collaborator.

You are asked to implement a Python module that recommends available meeting slots within a
defined working window.

The system must:
  • Accept working hours (start and end time).
  • Accept a list of existing busy intervals.
  • Accept a required meeting duration.
  • Accept an optional buffer time between meetings.
  • Optionally restrict suggestions to a candidate time window.
  • Return chronologically ordered appointment slots that satisfy all constraints.

The system must ensure that:
  • Suggested slots fall within working hours.
  • Suggested slots do not overlap busy intervals.
  • Buffer time is respected when evaluating availability.
  • Output ordering is deterministic under identical inputs.

The module must preserve the following invariants:
  • Returned slots must be at least as long as the required duration.
  • No returned slot may violate buffer constraints.
  • The returned list must reflect the current system state.

The system must correctly handle non-trivial scenarios such as:
  • Adjacent busy intervals.
  • Very small gaps between meetings.
  • Buffers eliminating otherwise valid availability.
  • Overlapping or unsorted busy intervals.
  • A meeting duration longer than any available gap.
  • No availability within the working window.

Output:
  The output consists of the next N valid appointment suggestions in chronological order.
  Behavior must be deterministic under ties (if any).

See the lab handout for full requirements.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta, time
from typing import List, Optional, Tuple


# ---------------- Data Models ----------------

@dataclass(frozen=True)
class TimeWindow:
    """
    A daily time window.
    Assumption (unless stated otherwise in handout): non-wrapping window where start < end.
    """
    start: time
    end: time


@dataclass(frozen=True)
class BusyInterval:
    """
    A busy interval on the given day.
    Invariant: start < end
    """
    start: time
    end: time


@dataclass(frozen=True)
class Slot:
    """
    A recommended appointment slot.

    start_time is a time-of-day within the working window.
    Deterministic ordering: sort by start_time ascending.
    """
    start_time: time


class InfeasibleSchedule(Exception):
    """Raised when no valid slots can be produced (if required by handout)."""
    pass


# ---------------- Helpers ----------------

_STEP_MINUTES = 5  # We generate possible meeting starts every 5 minutes


def _combine_date_and_time(day: date, t: time) -> datetime:
    """
    Combining a date and a time into a single datetime object.
    This makes it easier to perform comparisons and arithmetic.
    """
    return datetime.combine(day, t)


def _get_overlap(a: Tuple[datetime, datetime], b: Tuple[datetime, datetime]) -> Optional[Tuple[datetime, datetime]]:
    """
    Return the overlapping portion of two time windows.

    Each window is treated as half-open: [start, end).
    If they do not overlap, return None.
    """
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    return (s, e) if e > s else None


def _round_up_to_step(dt: datetime, step: timedelta) -> datetime:
    """
    Round a datetime up to the next valid step boundary.

    For example, if we use 5-minute steps and the time is 9:02,
    this function will return 9:05.

    This ensures deterministic slot generation.
    """
    midnight = datetime(dt.year, dt.month, dt.day, 0, 0, 0)
    diff = dt - midnight
    step_seconds = int(step.total_seconds())
    if step_seconds <= 0:
        return dt
    seconds = int(diff.total_seconds())
    rem = seconds % step_seconds
    if rem == 0:
        return dt
    return dt + timedelta(seconds=(step_seconds - rem))


def _merge_overlapping_intervals(intervals: List[Tuple[datetime, datetime]]) -> List[Tuple[datetime, datetime]]:
    """
    Sort intervals and merge any that overlap or directly touch.

    This ensures we correctly handle:
      - overlapping busy intervals
      - unsorted inputs
      - adjacent intervals
    """
    if not intervals:
        return []
    intervals.sort(key=lambda x: (x[0], x[1]))
    merged: List[Tuple[datetime, datetime]] = []
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        # If the intervals overlap (or touch), we extend the current interval
        if s <= cur_e:  # overlap
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged



# ---------------- Core Function ----------------

def suggest_slots(
    day: date,
    working_hours: TimeWindow,
    busy_intervals: List[BusyInterval],
    duration: timedelta,
    n: int,
    buffer: timedelta = timedelta(0),
    candidate_window: Optional[TimeWindow] = None
) -> List[Slot]:
    """
    Suggest up to the next n valid appointment slots (start times) for the given day.

    Args:
        day: the calendar day for which to suggest slots.
        working_hours: the allowed working window for meetings (start < end).
        busy_intervals: list of busy time intervals (may be overlapping / unsorted).
        duration: required meeting length (must be > 0).
        n: maximum number of slot suggestions to return (n >= 0).
        buffer: optional buffer time required between meetings (buffer >= 0).
        candidate_window: optional extra restriction on suggestions (must lie within this window too).

    Returns:
        A list of Slot objects, sorted by start_time ascending, deterministic under identical inputs.
        If no suitable time slots are available, return an empty list.

    Notes:
        - Suggested slots must fall within working_hours (and candidate_window if provided).
        - Suggested slots must not overlap busy_intervals, considering buffer time.
        - You are free to choose internal representation; inputs use time-of-day.
        - See lab handout for required slot granularity (e.g., 5-min/15-min steps), if any.
    """

    ##################################################################
    # TODO: Implement as per lab handout requirements and constraints.
    ##################################################################
    
    # Input validation and invariants
    if n <= 0:
        return []
    if duration is None or duration <= timedelta(0):
        return []
    if buffer is None or buffer < timedelta(0):
        buffer = timedelta(0)

    # Building the available allowed-range from working hours (+ candidate window if provided)
    wh = (_combine_date_and_time(day, working_hours.start), _combine_date_and_time(day, working_hours.end))
    if wh[1] <= wh[0]:
        return []

    if candidate_window is not None:
        cw = (_combine_date_and_time(day, candidate_window.start), _combine_date_and_time(day, candidate_window.end))
        if cw[1] <= cw[0]:
            return []
        inter = _get_overlap(wh, cw)
        if inter is None:
            return []
        allowed_range = inter
    else:
        allowed_range = wh

    allowed_start, allowed_end = allowed_range

    # Expanding busy intervals by buffer and collecting as datetimes
    expanded: List[Tuple[datetime, datetime]] = []
    for bi in busy_intervals:
        s = _combine_date_and_time(day, bi.start)
        e = _combine_date_and_time(day, bi.end)
        if e <= s:
            continue
        s2 = s - buffer
        e2 = e + buffer
        expanded.append((s2, e2))

    merged_busy = _merge_overlapping_intervals(expanded)

    # Computing free segments within allowed_range [allowed_start, allowed_end)
    free_segments: List[Tuple[datetime, datetime]] = []
    cursor = allowed_start

    for bs, be in merged_busy:
        # Skips intervals entirely before allowed_range start
        if be <= allowed_start:
            continue
        # Stops if busy begins after allowed_range end
        if bs >= allowed_end:
            break

        seg_start = cursor
        seg_end = min(bs, allowed_end)
        if seg_end > seg_start:
            free_segments.append((seg_start, seg_end))

        cursor = max(cursor, be)
        if cursor >= allowed_end:
            break

    if cursor < allowed_end:
        free_segments.append((cursor, allowed_end))

    # Counting starts from segments in 5-minute steps
    step = timedelta(minutes=_STEP_MINUTES)
    results: List[Slot] = []

    for fs, fe in free_segments:
        last_start = fe - duration
        if last_start < fs:
            continue

        t = _round_up_to_step(fs, step)
        while t <= last_start and len(results) < n:
            results.append(Slot(start_time=t.time()))
            t += step

        if len(results) >= n:
            break

    # The results are already in chronological order based on how we generate them.
    # Thus, sorting again simply guarantees this invariant.
    results.sort(key=lambda s: (s.start_time.hour, s.start_time.minute, s.start_time.second))
    return results
