import re
import time
import requests
from bs4 import BeautifulSoup

# Headers to mimic a real browser - reduces chance of being blocked
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

# Tags that usually contain job description content
# LinkedIn selectors discovered via live inspection 2026-03
JD_CONTENT_SELECTORS = [
    # LinkedIn - primary (most content)
    "div.decorated-job-posting__details",
    "section.core-section-container.my-3.description",
    "div.description__text--rich",
    "div.description__text",
    "section.show-more-less-html",
    "div.show-more-less-html__markup",
    # Adzuna detail page
    "div.adp-body",
    "div.job-ad",
    "div.job-description-wrapper",
    # Naukri
    "div.job-desc",
    "div.dang-inner-html",
    "div.styles_JDC__dang-inner-html__h0K4t",
    # Indeed
    "div#jobDescriptionText",
    "div.jobsearch-jobDescriptionText",
    # Glassdoor
    "div.jobDescriptionContent",
    "div[class*='JobDetails']",
    # Shine / TimesJobs / other Indian portals
    "div.job-desc-container",
    "div.desc-container",
    "div#job-desc",
    # Generic fallbacks
    "div.job-description",
    "div.description",
    "section.job-description",
    "div[class*='description']",
    "div[class*='job-detail']",
]

# Tags to remove (noise)
NOISE_TAGS = [
    "script",
    "style",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "button",
    "iframe",
    "noscript",
    "svg",
    "img",
    "figure",
    "advertisement",
]


def _clean_text(text: str) -> str:
    """Remove extra whitespace and normalize line breaks."""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\u00a0", " ", text)  # non-breaking space
    return text.strip()


def _extract_from_soup(soup: BeautifulSoup) -> str:
    """
    Try known selectors first, fall back to full body text.
    Returns the longest meaningful text block found.
    Priority: known selectors > largest div > body text
    """
    # remove noise tags
    for tag in soup.find_all(NOISE_TAGS):
        tag.decompose()

    # collect all candidate texts from known selectors
    candidates = []

    for selector in JD_CONTENT_SELECTORS:
        try:
            els = soup.select(selector)  # get ALL matching elements
            for el in els:
                text = el.get_text(separator="\n")
                cleaned = _clean_text(text)
                if len(cleaned) > 200:
                    candidates.append(cleaned)
        except Exception:
            continue

    if candidates:
        # return the longest - most complete JD content
        best = max(candidates, key=len)
        if len(best) > 200:
            return best

    # fallback: find the largest meaningful text block on the page
    page_candidates = []
    for tag in soup.find_all(["div", "section", "article"]):
        text = tag.get_text(separator="\n")
        cleaned = _clean_text(text)
        if 300 < len(cleaned) < 50000:  # skip tiny + huge blocks
            page_candidates.append(cleaned)

    if page_candidates:
        return max(page_candidates, key=len)

    # last resort: full body text
    body = soup.find("body")
    return _clean_text(body.get_text(separator="\n")) if body else ""


def fetch_full_jd(url: str, timeout: int = 10) -> str:
    """
    Fetch the full job description text from a URL.

    Returns:
        Full JD text string (can be 1000-5000 chars for a real JD)
        Empty string if fetch fails or content is too short to be useful

    Handles:
        - Timeouts gracefully
        - Anti-scraping redirects (returns empty rather than crash)
        - Adzuna redirect URLs (follows redirects automatically)
    """
    if not url or url.strip() == "":
        return ""

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=timeout,
            allow_redirects=True,
        )

        # some sites return 403/429 for bots - return empty gracefully
        if response.status_code in (403, 429, 503):
            print(
                f"    JD fetch blocked ({response.status_code}): {url[:60]}..."
            )
            return ""

        if response.status_code != 200:
            return ""

        # parse HTML
        soup = BeautifulSoup(response.text, "html.parser")
        text = _extract_from_soup(soup)

        if len(text) < 100:
            return ""

        # cap at 4000 chars - enough for complete JD, not too much for Gemini
        return text[:4000]

    except requests.exceptions.Timeout:
        print(f"    JD fetch timeout: {url[:60]}...")
        return ""
    except Exception as e:
        print(f"    JD fetch error: {e}")
        return ""


def fetch_full_jds_batch(
    jobs: list,
    max_jobs: int = 20,
    delay: float = 0.5,
) -> dict[str, str]:
    """
    Fetch full JD text for a list of JobListing objects.

    Args:
        jobs:     list of JobListing objects with .job_id and .apply_url
        max_jobs: maximum number of jobs to fetch (default 20)
        delay:    seconds to wait between requests (be polite to servers)

    Returns:
        dict mapping job_id -> full_jd_text
        Only includes jobs where fetch succeeded and text > 100 chars
    """
    results = {}
    fetch_count = 0

    for job in jobs[:max_jobs]:
        if not job.apply_url:
            continue

        print(
            f"    Fetching full JD [{fetch_count+1}/{min(len(jobs), max_jobs)}]: "
            f"{job.title[:40]}..."
        )

        text = fetch_full_jd(job.apply_url)

        if text:
            results[job.job_id] = text
            print(f"    Got {len(text)} chars")
        else:
            print("    No content - keeping Adzuna snippet")

        fetch_count += 1
        time.sleep(delay)  # polite delay between requests

    success_rate = len(results) / max(fetch_count, 1) * 100
    print(
        f"\n    JD fetch complete: {len(results)}/{fetch_count} succeeded "
        f"({success_rate:.0f}% success rate)"
    )

    return results
