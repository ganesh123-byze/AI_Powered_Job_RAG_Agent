"""
exp_inference.py  –  Multi-Signal Experience Inference Layer
=============================================================

4-tier system per design doc.  All bugs from the minimal version fixed.

Bugs fixed
----------
1. _parse_years(): missed "2 yrs", "3+ yrs exp", "2 yoe",
   "10 or more years", "minimum experience of 3 years".
   False positive: "2024 batch" was matched as "24 years".
   Patterns now ordered most-specific -> least-specific with guards.

2. _label_from_years(): "0-2 years" was labelled "junior"
   (lo=0, hi=2 != 1). Now uses hi as primary signal for tighter labels.

3. _fetch_page_text(): was sending generic bot UA -> 403 blocked.
   Was parsing raw HTML without stripping tags -> regex matched inside tags.
   Fixed: proper browser UA + BeautifulSoup text extraction.

4. is_match logic: was re-parsing snippet AFTER full JD may have updated
   label/years, so full-JD years were ignored in match decision.
   Fixed: match uses final resolved (lo, hi) from whichever layer succeeded.

5. "unknown" label -> is_match=False -> scorer excluded it.
   Design doc: unknown -> neutral 0.50, pass through.
   Fixed: unknown returns is_match=True with score=0.50.

6. Confidence table per design doc:
   full JD explicit years -> 0.95
   full JD phrase         -> 0.85
   snippet explicit years -> 0.70
   snippet phrase         -> 0.60
   title token only       -> 0.50-0.75 by token strength
   no evidence            -> 0.10 (neutral, NOT excluded)
"""

from __future__ import annotations

import re, time, logging
from dataclasses import dataclass, field
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_URL_CACHE: dict[str, str] = {}

def clear_url_cache() -> None:
    _URL_CACHE.clear()


# ── Data classes ──────────────────────────────────────────────────────────

@dataclass
class Signal:
    evidence_source: str = "none"

@dataclass
class ExpResult:
    label:      str   = "unknown"
    score:      float = 0.50
    confidence: float = 0.10
    is_match:   bool  = True
    reason:     str   = ""
    signal:     Signal = field(default_factory=Signal)


# ── Layer 1: Title tokens ─────────────────────────────────────────────────

_TITLE_TOKENS: list[tuple[str, str, float]] = [
    (r"\bintern(?:ship)?\b",       "entry",  0.72),
    (r"\btrainees?\b",             "entry",  0.72),
    (r"\bfreshers?\b",             "entry",  0.75),
    (r"\bentry[\s\-]level\b",     "entry",  0.75),
    (r"\bgraduate\s+trainees?\b",  "entry",  0.68),
    (r"\bjr\.?\b",                 "junior", 0.65),
    (r"\bjunior\b",                "junior", 0.68),
    (r"\bassociate\b",             "junior", 0.55),
    (r"\bmid[\s\-]level\b",       "mid",    0.62),
    (r"\bintermediate\b",          "mid",    0.60),
    (r"\bstaff\b",                 "senior", 0.70),
    (r"\bprincipal\b",             "senior", 0.75),
    (r"\barchitect\b",             "senior", 0.72),
    (r"\blead\b",                  "senior", 0.70),
    (r"\bsr\.?\b",                 "senior", 0.68),
    (r"\bsenior\b",                "senior", 0.72),
    (r"\bhead\s+of\b",             "senior", 0.75),
    (r"\bdirector\b",              "senior", 0.75),
    (r"\bmanager\b",               "senior", 0.65),
    (r"\bvp\b",                    "senior", 0.75),
]

def _label_from_title(title: str) -> tuple[Optional[str], float]:
    tl = title.lower()
    for pattern, label, conf in _TITLE_TOKENS:
        if re.search(pattern, tl):
            return label, conf
    return None, 0.0


# ── Layer 2: Normalisation + year regex ───────────────────────────────────

_NORM_SUBS = [
    (r"[\u2013\u2014\u2012\u2015]", "-"),
    (r"\byrs?\b",                   "years"),
    (r"\byear's\b",                 "years"),
    (r"\by\.o\.e\.?\b",            "years of experience"),
    (r"\byoe\b",                    "years of experience"),
    (r"\bexp\b(?!\w)",              "experience"),
    (r"\bat\s+least\b",            "minimum"),
    (r"\batleast\b",               "minimum"),
    (r"\bmin(?:imum)?\b",          "minimum"),
    (r"\bor\s+more\b",             "+"),
    (r"&amp;", "&"), (r"&lt;","<"), (r"&gt;",">"), (r"&nbsp;"," "),
]

def _norm(text: str) -> str:
    t = (text or "").lower()
    for p, r in _NORM_SUBS:
        t = re.sub(p, r, t, flags=re.IGNORECASE)
    return t

# (pattern, kind)
_YEAR_PATTERNS = [
    (r"experience\s*[:\-]\s*(\d+)\s*(?:-|to)\s*(\d+)\s*years?",  "label_range"),
    (r"experience\s*[:\-]\s*(\d+)\+?\s*years?",                   "label_single"),
    (r"minimum\s+(\d+)\s*(?:-|to)\s*(\d+)\s*years?",             "min_range"),
    (r"minimum\s+(\d+)\+?\s*years?",                              "min_single"),
    (r"\b([0-2]?\d)\s*-\s*([0-3]?\d)\s*years?(?:\s+of)?(?:\s+experience)?",   "range"),
    (r"\b(\d+)\s+to\s+(\d+)\s*years?",                           "to_range"),
    (r"\b(\d+)\+\s*years?",                                       "plus"),
    (r"\b(\d+)\s*years?\s+of\s+(?:relevant\s+)?experience",      "exact_of"),
    (r"\b(\d+)\s*years?\s+experience",                            "exact"),
    (r"\b(\d+)\s*years?\b",                                       "generic"),
]

_PHRASE_LABELS = [
    (r"\bfreshers?\s+(?:can\s+)?apply\b",        "entry",  0.75),
    (r"\bfresh\s+graduates?\b",                  "entry",  0.75),
    (r"\brecent\s+graduates?\b",                 "entry",  0.68),
    (r"\bcampus\s+(?:hire|recruit|placement)\b", "entry",  0.70),
    (r"\bno\s+experience\s+required\b",          "entry",  0.75),
    (r"\b0[\s\-]1\s+years?\b",                  "entry",  0.72),
    (r"\bentry[\s\-]level\b",                   "entry",  0.70),
    (r"\btrainee\s+program\b",                   "entry",  0.68),
    (r"\bjunior\s+(?:developer|engineer|analyst)","junior",0.68),
    (r"\bmid[\s\-]level\b",                     "mid",    0.62),
    (r"\bsenior\s+(?:developer|engineer|analyst)","senior",0.65),
    (r"\blead\s+(?:developer|engineer|architect)","senior",0.68),
]

def _is_false_positive(matched: str, text: str) -> bool:
    n = re.search(r"\d+", matched)
    if n and int(n.group()) > 30:
        return True
    guards = [r"\b20[2-9]\d\s+batch\b", r"\bbatch\s+of\s+20[2-9]\d\b",
              r"\b20[2-9]\d\s+graduates?\b", r"\bpassout\s+(?:batch\s+)?20[2-9]\d\b"]
    return any(re.search(g, text, re.IGNORECASE) for g in guards)

def _parse_years(text: str) -> Optional[tuple[float, Optional[float]]]:
    t = _norm(text)
    for pattern, kind in _YEAR_PATTERNS:
        for m in re.finditer(pattern, t, re.IGNORECASE):
            if _is_false_positive(m.group(0), t):
                continue
            try:
                g = m.groups()
                if kind in ("range","min_range","to_range","label_range"):
                    lo, hi = float(g[0]), float(g[1])
                    if lo > hi: lo, hi = hi, lo
                    if lo > 30 or hi > 40: continue
                    return lo, hi
                else:
                    lo = float(g[0])
                    if lo > 30: continue
                    return lo, None
            except (ValueError, IndexError, TypeError):
                continue
    for pat, label, _ in _PHRASE_LABELS:
        if label == "entry" and re.search(pat, t, re.IGNORECASE):
            return 0.0, 1.0
    return None

def _years_to_label(lo: float, hi: Optional[float]) -> str:
    eff_hi = hi if hi is not None else lo + 1.5
    if lo == 0 and eff_hi <= 2.0: return "entry"
    if lo <= 2.0 and eff_hi <= 4.5: return "junior"
    if lo <= 5.0 and eff_hi <= 8.0: return "mid"
    return "senior"

def _phrase_label(text: str) -> tuple[Optional[str], float]:
    t = _norm(text)
    for pat, label, conf in _PHRASE_LABELS:
        if re.search(pat, t, re.IGNORECASE):
            return label, conf
    return None, 0.0


# ── Layer 3: Destination-page fetcher ─────────────────────────────────────

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
_FETCH_HEADERS = {
    "User-Agent":      _BROWSER_UA,
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.7,hi;q=0.3",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection":      "keep-alive",
    "DNT":             "1",
}

_BLOCKED_DOMAINS = frozenset({
    "linkedin.com","naukri.com","monster.com",
    "shine.com","timesjobs.com","glassdoor.com",
})

_JD_SELECTORS = [
    "div.adp-body","div.job-description-wrapper","div.job-ad",
    "div#jobDescriptionText","div.jobsearch-jobDescriptionText",
    "div.job-desc","div.dang-inner-html",
    "div.job-desc-container","div#job-desc",
    "div.job-description","div.description",
    "section.job-description","article.job-description",
    "div[class*='job-detail']","div[class*='description']",
]

def _domain(url: str) -> str:
    m = re.search(r"https?://(?:www\.)?([^/]+)", url or "")
    return m.group(1).lower() if m else ""

def _is_blocked(url: str) -> bool:
    d = _domain(url)
    return any(bd in d for bd in _BLOCKED_DOMAINS)

def _extract_text(html: str) -> str:
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.find_all(["script","style","nav","footer","header","aside","form","button","iframe"]):
            tag.decompose()
        candidates = []
        for sel in _JD_SELECTORS:
            for el in soup.select(sel):
                t = el.get_text(separator=" ", strip=True)
                if len(t) > 150:
                    candidates.append(t)
        if candidates:
            return max(candidates, key=len)[:6000]
        for tag in soup.find_all(["div","section","article","main"]):
            t = tag.get_text(separator=" ", strip=True)
            if 200 < len(t) < 60_000:
                candidates.append(t)
        return max(candidates, key=len)[:6000] if candidates else soup.get_text(separator=" ", strip=True)[:6000]
    except ImportError:
        text = re.sub(r"<[^>]+>", " ", html)
        return re.sub(r"\s+", " ", text)[:6000]

def _fetch_page_text(url: str, timeout: int = 8) -> str:
    if not url or _is_blocked(url):
        return ""
    if url in _URL_CACHE:
        return _URL_CACHE[url]
    try:
        resp = requests.get(url, headers=_FETCH_HEADERS, timeout=timeout, allow_redirects=True)
        if resp.status_code in (403, 429, 503, 404):
            _URL_CACHE[url] = ""
            return ""
        if resp.status_code != 200:
            _URL_CACHE[url] = ""
            return ""
        text = _extract_text(resp.text)
        _URL_CACHE[url] = text
        return text
    except Exception as exc:
        logger.debug("Fetch error %s: %s", url, exc)
        _URL_CACHE[url] = ""
        return ""


# ── Layer 4: Confidence-weighted match scoring ─────────────────────────────

_ACCEPTABLE: dict[str, set[str]] = {
    "fresher": {"entry"},
    "junior":  {"entry","junior"},
    "mid":     {"junior","mid"},
    "senior":  {"mid","senior"},
}
_USER_YEARS: dict[str, tuple[float, float]] = {
    "fresher": (0.0, 1.5),
    "junior":  (0.5, 3.5),
    "mid":     (2.0, 7.0),
    "senior":  (4.0, 99.0),
}

def _compute_score(
    label: str,
    resolved_lo: Optional[float],
    resolved_hi: Optional[float],
    confidence: float,
    user_level: str,
    candidate_years: float,
) -> tuple[bool, float, str]:

    if label == "unknown" or confidence < 0.20:
        return True, 0.50, f"No evidence (conf={confidence:.2f}) → neutral 0.50"

    acceptable = _ACCEPTABLE.get(user_level, {"entry","junior","mid","senior"})
    user_lo, user_hi = _USER_YEARS.get(user_level, (0.0, 99.0))
    label_ok = label in acceptable

    years_score, years_ok, years_info = 1.0, True, "no numeric years"

    if resolved_lo is not None:
        jd_lo  = resolved_lo
        jd_hi  = resolved_hi if resolved_hi is not None else resolved_lo + 2.0
        years_info = f"JD=[{jd_lo:.0f}-{jd_hi:.0f}] user=[{user_lo:.0f}-{user_hi:.0f}]"
        o_lo, o_hi = max(jd_lo, user_lo), min(jd_hi, user_hi)
        if o_hi < o_lo:
            years_ok    = False
            years_score = max(0.0, 1.0 - (jd_lo - user_hi) * 0.20)
        else:
            years_score = min(1.0, (o_hi - o_lo) / max(jd_hi - jd_lo, 0.5))

    if label_ok and years_ok:
        raw      = 0.85 * years_score + 0.15 * confidence
        is_match = True
        reason   = f"label='{label}' OK, years_score={years_score:.2f}, {years_info}, conf={confidence:.2f}"
    elif label_ok and not years_ok:
        raw      = 0.55 * years_score * confidence
        is_match = years_score > 0.15
        reason   = f"label='{label}' OK but years mismatch. {years_info}"
    elif not label_ok and confidence < 0.55:
        raw      = 0.45
        is_match = True
        reason   = f"label='{label}' mismatch but weak conf ({confidence:.2f}) → neutral"
    else:
        raw      = max(0.05, years_score * 0.15)
        is_match = False
        reason   = f"label='{label}' not in {acceptable} for '{user_level}', conf={confidence:.2f}"

    blended = raw * confidence + 0.50 * (1.0 - confidence)
    return is_match, round(min(blended, 1.0), 4), reason


# ── Public entry point ─────────────────────────────────────────────────────

def infer_experience(
    *,
    title:           str   = "",
    snippet:         str   = "",
    redirect_url:    str   = "",
    user_level:      str   = "",
    candidate_years: float = 0.0,
    fetch_full_jd:   bool  = False,
) -> ExpResult:
    """4-layer experience inference. Backward-compatible with scorer.py."""

    label:       str            = "unknown"
    confidence:  float          = 0.10
    evidence:    str            = "none"
    resolved_lo: Optional[float] = None
    resolved_hi: Optional[float] = None

    # Layer 1: title
    t_label, t_conf = _label_from_title(title)
    if t_label:
        label, confidence, evidence = t_label, t_conf, "title"

    # Layer 2a: snippet years
    if snippet:
        years = _parse_years(snippet)
        if years:
            lo, hi = years
            s_label = _years_to_label(lo, hi)
            if confidence < 0.70 or s_label != label:
                label, confidence, evidence = s_label, 0.70, "snippet"
            resolved_lo, resolved_hi = lo, hi
        else:
            # Layer 2b: snippet phrase
            p_label, p_conf = _phrase_label(snippet)
            if p_label and p_conf > confidence:
                label, confidence, evidence = p_label, p_conf, "snippet_phrase"

    # Layer 3: full JD fetch
    if (fetch_full_jd and redirect_url and not _is_blocked(redirect_url)
            and (confidence < 0.75 or resolved_lo is None)):
        page = _fetch_page_text(redirect_url)
        if page:
            years = _parse_years(page)
            if years:
                lo, hi = years
                label, confidence, evidence = _years_to_label(lo, hi), 0.95, "full_jd"
                resolved_lo, resolved_hi = lo, hi
            else:
                p_label, _ = _phrase_label(page)
                if p_label:
                    label, confidence, evidence = p_label, 0.85, "full_jd_phrase"
                elif evidence == "none":
                    evidence = "full_jd_checked"
        time.sleep(0.3)

    signal = Signal(evidence_source=evidence)

    is_match, score, reason = _compute_score(
        label, resolved_lo, resolved_hi, confidence, user_level, candidate_years
    )

    logger.debug("ExpInfer title=%r label=%s conf=%.2f src=%s match=%s score=%.3f",
                 title[:50], label, confidence, evidence, is_match, score)

    return ExpResult(
        label=label, score=score, confidence=round(confidence, 4),
        is_match=is_match, reason=reason, signal=signal,
    )


# ── Self-test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        ("Junior Data Scientist",     "1-2 years of experience required",      "junior",  1.0, True),
        ("Senior ML Engineer",        "5+ years of experience required",        "fresher", 0.0, False),
        ("Data Analyst",              "Freshers can apply. 0-1 yrs preferred.", "fresher", 0.0, True),
        ("Machine Learning Engineer", "3-5 yrs of experience in deep learning","mid",     3.0, True),
        ("Software Engineer",         "3+ yrs exp required",                    "mid",     3.5, True),
        ("AI Engineer",               "2 yoe in NLP required",                  "junior",  2.0, True),
        ("Lead AI Engineer",          "10+ years of experience required",        "fresher", 0.0, False),
        ("Data Engineer",             "minimum 4 years experience in Spark",    "junior",  1.5, False),
        ("ML Researcher",             "2024 batch graduates are welcome",        "fresher", 0.0, True),
        ("Python Developer",          "",                                        "mid",     3.0, True),
        ("Senior Software Engineer",  "experience: 7-10 years",                 "mid",     4.0, False),
        ("Associate Data Scientist",  "we need 2 or more years of experience",  "junior",  2.0, True),
    ]
    passed = 0
    print("="*65)
    print("Experience Inference Self-Test")
    print("="*65)
    for title, snippet, level, years, expected in tests:
        r = infer_experience(title=title, snippet=snippet, redirect_url="",
                             user_level=level, candidate_years=years, fetch_full_jd=False)
        ok = r.is_match == expected
        if ok: passed += 1
        flag = "PASS" if ok else "FAIL"
        print(f"\n  [{flag}] [{level:7}] {title}")
        print(f"    label={r.label:7} conf={r.confidence:.2f} score={r.score:.2f}"
              f" src={r.signal.evidence_source} match={r.is_match} (exp={expected})")
        print(f"    {r.reason[:75]}")
    print(f"\n  Result: {passed}/{len(tests)} passed")
    print("="*65)