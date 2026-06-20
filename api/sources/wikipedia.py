"""
Wikipedia source — fetches a short plain-language summary for a domain or concept.
Used to give users accessible context about analogy domains.
"""

import json
import urllib.request
import urllib.parse


def get_summary(topic: str) -> dict:
    encoded = urllib.parse.quote(topic)
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ProblemTransferEngine/1.0"})
        with urllib.request.urlopen(req, timeout=8) as response:
            data = json.loads(response.read())
        return {
            "title": data.get("title", ""),
            "extract": data.get("extract", "")[:400],
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
        }
    except Exception:
        return {}
