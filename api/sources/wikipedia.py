"""
Wikipedia source — fetches a plain-language summary for a domain or concept.
Never raises — returns empty dict on failure so callers can degrade gracefully.
"""

import json
import urllib.request
import urllib.parse


def get_summary(topic: str) -> dict:
    encoded = urllib.parse.quote(topic.replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ProblemTransferEngine/1.0"})
        with urllib.request.urlopen(req, timeout=8) as response:
            data = json.loads(response.read())
        if data.get("type") == "disambiguation":
            return {}
        return {
            "title": data.get("title", ""),
            "extract": data.get("extract", "")[:2000],
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            "thumbnail": (data.get("thumbnail") or {}).get("source", ""),
        }
    except Exception:
        return {}
