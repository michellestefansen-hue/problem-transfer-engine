"""
OpenAlex source — searches for academic papers by query.

Current status: integrated but results are best used as supplementary context.
Keyword search returns high-citation papers that may not always be domain-specific.

To add a new search strategy, add a function here and import it in index.py.
"""

import json
import urllib.request
import urllib.parse


def search(query: str, max_results: int = 3) -> list:
    encoded = urllib.parse.quote(query)
    url = (
        f"https://api.openalex.org/works"
        f"?search={encoded}"
        f"&filter=is_oa:true"
        f"&sort=relevance_score:desc"
        f"&per-page={max_results}"
        f"&select=title,doi,publication_year,cited_by_count,primary_location,concepts"
        f"&mailto=problem-transfer-engine@example.com"
    )
    try:
        with urllib.request.urlopen(url, timeout=8) as response:
            data = json.loads(response.read())
        papers = []
        for work in data.get("results", []):
            source = work.get("primary_location") or {}
            source_name = (source.get("source") or {}).get("display_name", "")
            concepts = [c["display_name"] for c in work.get("concepts", [])[:3]]
            papers.append({
                "title": work.get("title", ""),
                "year": work.get("publication_year"),
                "citations": work.get("cited_by_count", 0),
                "source": source_name,
                "doi": work.get("doi", ""),
                "concepts": concepts,
            })
        return papers
    except Exception:
        return []
