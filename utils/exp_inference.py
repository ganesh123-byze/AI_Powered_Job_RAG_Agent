"""
Minimal experience inference utilities.

Provides:
 - `infer_experience(...)` : lightweight layered inference (title/snippet/optional full page)
 - `clear_url_cache()` : clear internal landing-page cache

This module implements a conservative, dependency-free inference used as a safe
fallback so the rest of the pipeline can run while we iteratively improve the
full-tiered solution (landing-page parsers / headless browser fetchers etc.).
"""

from __future__ import annotations

import re
import requests
from dataclasses import dataclass
from typing import Optional

# Simple in-memory cache for fetched pages to avoid repeated network calls
_URL_CACHE: dict[str, str] = {}


@dataclass
class Signal:
    evidence_source: str = "none"


@dataclass
class ExpResult:
    label: str
    score: float
    confidence: float
    is_match: bool
    reason: str
    signal: Signal


def clear_url_cache() -> None:
    _URL_CACHE.clear()


def _parse_years_from_text(text: str) -> Optional[tuple[int, Optional[int], str]]:
    t = (text or "").lower()
    t = re.sub(r"[\u2013\u2014–—]", "-", t)

    # explicit ranges: 3-5 years, 3 to 5 years
    m = re.search(r"(\d+)\s*(?:-|to)\s*(\d+)\s*years?", t)
    if m:
        lo = int(m.group(1))
        hi = int(m.group(2))
        return lo, hi, "range"

    # explicit single: 3+ years, 3+ years of experience
    m = re.search(r"(\d+)\+?\s*years?", t)
    if m:
        lo = int(m.group(1))
        return lo, None, "single"

    # entry, fresher, graduate
    if any(k in t for k in ("entry level", "entry-level", "fresher", "graduate", "intern")):
        return 0, 1, "entry"

    return None


def _label_from_years(lo: Optional[int], hi: Optional[int]) -> str:
    if lo is None and hi is None:
        return "unknown"
    if lo == 0 and (hi == 1 or hi is None):
        return "entry"
    low = lo or 0
    if low <= 1:
        return "junior"
    if low <= 4:
        return "mid"
    return "senior"


def _fetch_landing_page(url: str, timeout: int = 6) -> Optional[str]:
    if not url:
        return None
    if url in _URL_CACHE:
        return _URL_CACHE[url]
    try:
        headers = {"User-Agent": "JobRAG-Agent/1.0 (+https://example.org)"}
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code == 200 and r.text:
            _URL_CACHE[url] = r.text
            return r.text
    except Exception:
        return None
    return None


def infer_experience(
    *,
    title: str = "",
    snippet: str = "",
    redirect_url: str = "",
    user_level: str = "",
    candidate_years: float = 0.0,
    fetch_full_jd: bool = False,
) -> ExpResult:
    """Conservative, layered experience inference.

    Returns an ExpResult with label, numeric score (0-1), confidence, is_match,
    a short human-readable reason, and a `Signal` describing evidence source.
    """

    # 1) Try snippet
    parsed = _parse_years_from_text(snippet)
    if parsed:
        lo, hi, kind = parsed
        label = _label_from_years(lo, hi)
        confidence = 0.70
        evidence = "snippet"
        reason = f"Found explicit phrase in snippet ({kind})."
    else:
        # 2) Infer from title tokens
        tl = (title or "").lower()
        title_tokens = ["intern", "fresher", "junior", "associate", "mid", "senior", "lead", "principal"]
        found = next((t for t in title_tokens if t in tl), None)
        if found:
            if found in ("intern", "fresher"):
                label = "entry"
            elif found in ("junior", "associate"):
                label = "junior"
            elif found in ("mid",):
                label = "mid"
            else:
                label = "senior"
            confidence = 0.45
            evidence = "title"
            reason = f"Inferred from title token '{found}'."
        else:
            label = "unknown"
            confidence = 0.10
            evidence = "none"
            reason = "No explicit years or title clue found."

    # 3) Optionally fetch landing page if requested and evidence is weak
    signal = Signal(evidence_source=evidence)
    if fetch_full_jd and confidence < 0.80 and redirect_url:
        page = _fetch_landing_page(redirect_url)
        if page:
            parsed2 = _parse_years_from_text(page)
            if parsed2:
                lo2, hi2, kind2 = parsed2
                label = _label_from_years(lo2, hi2)
                confidence = 0.95
                signal.evidence_source = "full_jd"
                reason = f"Found explicit phrase on landing page ({kind2})."
            else:
                # still update evidence to indicate landing page was checked
                signal.evidence_source = "full_jd_checked"

    # 4) Determine match vs candidate years
    # For unknown labels, be conservative
    is_match = True
    if label == "unknown":
        is_match = False
    else:
        # approximate numeric check when possible
        parsed_snip = _parse_years_from_text(snippet)
        if parsed_snip:
            lo3, hi3, _ = parsed_snip
            if lo3 is not None:
                if hi3 is None:
                    is_match = candidate_years >= lo3
                else:
                    is_match = lo3 <= candidate_years <= hi3

    # 5) Compute a simple numeric score used by the scorer
    base = 0.5
    if is_match:
        base = 0.9
    score = round(base * confidence, 4)

    return ExpResult(
        label=label,
        score=score,
        confidence=round(confidence, 4),
        is_match=is_match,
        reason=reason,
        signal=signal,
    )
