from datetime import datetime, timezone
from config import FRESHNESS_SCORES


def get_freshness_bucket(posted_date_str: str) -> str:
    """
    Convert a posted date string (YYYY-MM-DD or ISO format)
    into a freshness bucket: today / this_week / this_month / older
    """
    try:
        posted_date_str = posted_date_str[:10]
        posted = datetime.strptime(posted_date_str, "%Y-%m-%d")
        posted = posted.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        days = (now - posted).days

        if days <= 1:
            return "today"
        elif days <= 7:
            return "this_week"
        elif days <= 30:
            return "this_month"
        else:
            return "older"

    except Exception:
        return "older"


def get_freshness_score(posted_date_str: str) -> float:
    """Convert a posted date string into a float score 0.0 - 1.0"""
    bucket = get_freshness_bucket(posted_date_str)
    return FRESHNESS_SCORES.get(bucket, 0.10)


def format_posted_date(posted_date_str: str) -> str:
    """Return a human readable label: 'Today', '3 days ago', '2 weeks ago' etc."""
    try:
        posted_date_str = posted_date_str[:10]
        posted = datetime.strptime(posted_date_str, "%Y-%m-%d")
        posted = posted.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        days = (now - posted).days

        if days == 0:
            return "Today"
        elif days == 1:
            return "Yesterday"
        elif days < 7:
            return f"{days} days ago"
        elif days < 14:
            return "1 week ago"
        elif days < 30:
            return f"{days // 7} weeks ago"
        elif days < 60:
            return "1 month ago"
        else:
            return f"{days // 30} months ago"

    except Exception:
        return "Unknown"
