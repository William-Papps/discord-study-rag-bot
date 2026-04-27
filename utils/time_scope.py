from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True)
class TimeScope:
    start: datetime
    end: datetime
    label: str


_RANGE_PATTERNS = [
    re.compile(
        r"\b(?:from|between)\s+(\d{4}-\d{2}-\d{2})\s+(?:to|and)\s+(\d{4}-\d{2}-\d{2})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(\d{4}-\d{2}-\d{2})\s+(?:to|-)\s+(\d{4}-\d{2}-\d{2})\b",
        re.IGNORECASE,
    ),
]
_SINGLE_DATE_PATTERN = re.compile(r"\bon\s+(\d{4}-\d{2}-\d{2})\b", re.IGNORECASE)


def parse_time_scope(question: str, now: datetime | None = None) -> TimeScope | None:
    text = question.strip().lower()
    if not text:
        return None

    base = (now or datetime.now().astimezone()).astimezone()

    for pattern in _RANGE_PATTERNS:
        match = pattern.search(text)
        if match:
            start_date = _parse_date(match.group(1), base.tzinfo)
            end_date = _parse_date(match.group(2), base.tzinfo)
            if start_date and end_date:
                start, end = sorted((start_date, end_date))
                return TimeScope(
                    start=_start_of_day(start),
                    end=_end_of_day(end),
                    label=f"{start.date().isoformat()} to {end.date().isoformat()}",
                )

    match = _SINGLE_DATE_PATTERN.search(text)
    if match:
        day = _parse_date(match.group(1), base.tzinfo)
        if day:
            return TimeScope(
                start=_start_of_day(day),
                end=_end_of_day(day),
                label=day.date().isoformat(),
            )

    if "yesterday" in text:
        day = base - timedelta(days=1)
        return TimeScope(start=_start_of_day(day), end=_end_of_day(day), label="yesterday")
    if "today" in text:
        return TimeScope(start=_start_of_day(base), end=_end_of_day(base), label="today")

    if "last week" in text:
        start = _start_of_week(base) - timedelta(days=7)
        end = start + timedelta(days=6)
        return TimeScope(start=_start_of_day(start), end=_end_of_day(end), label="last week")
    if "this week" in text:
        start = _start_of_week(base)
        end = start + timedelta(days=6)
        return TimeScope(start=_start_of_day(start), end=_end_of_day(end), label="this week")

    if "last month" in text:
        first_of_month = base.replace(day=1)
        last_of_previous = first_of_month - timedelta(days=1)
        start = last_of_previous.replace(day=1)
        return TimeScope(start=_start_of_day(start), end=_end_of_day(last_of_previous), label="last month")
    if "this month" in text:
        start = base.replace(day=1)
        next_month = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
        end = next_month - timedelta(days=1)
        return TimeScope(start=_start_of_day(start), end=_end_of_day(end), label="this month")

    if "last weekend" in text:
        start, end = _weekend_range(base, offset_weeks=1)
        return TimeScope(start=_start_of_day(start), end=_end_of_day(end), label="last weekend")
    if any(phrase in text for phrase in ("over the weekend", "over weekend", "this weekend", "the weekend")):
        start, end = _weekend_range(base, offset_weeks=0)
        return TimeScope(start=_start_of_day(start), end=_end_of_day(end), label="weekend")

    return None


def chunk_overlaps_time_scope(
    first_timestamp: str | None,
    last_timestamp: str | None,
    scope: TimeScope | None,
) -> bool:
    if scope is None:
        return True
    if not first_timestamp or not last_timestamp:
        return False
    first = _parse_iso(first_timestamp)
    last = _parse_iso(last_timestamp)
    if first is None or last is None:
        return False
    if last < first:
        first, last = last, first
    return last >= scope.start and first <= scope.end


def _parse_iso(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc).astimezone()
    return parsed.astimezone()


def _parse_date(value: str, tzinfo: timezone | None) -> datetime | None:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None
    return parsed.replace(tzinfo=tzinfo)


def _start_of_day(value: datetime) -> datetime:
    return value.replace(hour=0, minute=0, second=0, microsecond=0)


def _end_of_day(value: datetime) -> datetime:
    return value.replace(hour=23, minute=59, second=59, microsecond=999999)


def _start_of_week(value: datetime) -> datetime:
    return _start_of_day(value - timedelta(days=value.weekday()))


def _weekend_range(value: datetime, offset_weeks: int) -> tuple[datetime, datetime]:
    weekday = value.weekday()
    if weekday >= 5:
        saturday = _start_of_day(value - timedelta(days=weekday - 5))
    else:
        saturday = _start_of_day(value - timedelta(days=weekday + 2))
    saturday -= timedelta(days=7 * offset_weeks)
    sunday = saturday + timedelta(days=1)
    return saturday, sunday
