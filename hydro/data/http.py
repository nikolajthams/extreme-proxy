"""Minimal cached HTTP client (stdlib only) for the USGS services.

Every raw response is cached on disk so the whole pipeline is re-runnable
offline.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from hydro.config import CACHE_DIR

RAW_DIR = CACHE_DIR / "raw"
USER_AGENT = "extreme-proxy-research/0.1 (academic; contact via repo)"

# waterservices.usgs.gov intermittently 503s; nwis.waterservices.usgs.gov is the
# same service behind a different hostname and is used as a fallback.
HOST_FALLBACKS = {
    "waterservices.usgs.gov": "nwis.waterservices.usgs.gov",
}


def _cache_path(url: str) -> Path:
    digest = hashlib.sha256(url.encode()).hexdigest()[:32]
    return RAW_DIR / f"{digest}.gz"


def fetch(url: str, *, retries: int = 4, timeout: int = 180, refresh: bool = False) -> str:
    """GET `url`, returning the body as text. Cached forever on disk."""
    path = _cache_path(url)
    if path.exists() and not refresh:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return fh.read()

    urls = [url]
    for host, fallback in HOST_FALLBACKS.items():
        if host in url:
            urls.append(url.replace(host, fallback))

    last: Exception | None = None
    for attempt in range(retries):
        for candidate in urls:
            try:
                req = urllib.request.Request(candidate, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body = resp.read().decode("utf-8", errors="replace")
                path.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(path, "wt", encoding="utf-8") as fh:
                    fh.write(body)
                return body
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    # A genuine "no such site / no data" answer: cache the empty
                    # body so we do not re-ask.
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with gzip.open(path, "wt", encoding="utf-8") as fh:
                        fh.write("")
                    return ""
                last = exc
            except Exception as exc:  # noqa: BLE001 - network layer
                last = exc
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last}")


def fetch_json(url: str, **kwargs) -> dict:
    body = fetch(url, **kwargs)
    if not body.strip():
        return {}
    return json.loads(body)


def parse_rdb(body: str) -> list[dict[str, str]]:
    """Parse a USGS RDB (tab-separated, '#'-commented, second row = types)."""
    lines = [ln for ln in body.splitlines() if ln and not ln.startswith("#")]
    if len(lines) < 2:
        return []
    header = lines[0].split("\t")
    return [dict(zip(header, ln.split("\t"))) for ln in lines[2:] if ln.strip()]
