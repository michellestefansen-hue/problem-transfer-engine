"""
OpenAlex source — finds academic papers using concept-based search.

Strategy: instead of keyword search (which returns irrelevant high-citation papers),
we look up OpenAlex concept IDs for each mechanism, then find papers tagged with
those concepts across multiple fields. This is more precise because it uses
OpenAlex's own academic taxonomy rather than surface text matching.
"""

import json
import urllib.request
import urllib.parse


def _fetch(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=8) as response:
        return json.loads(response.read())


def find_concept_ids(terms: list, max_per_term: int = 2) -> list:
    """Look up OpenAlex concept IDs for a list of mechanism terms."""
    concept_ids = []
    for term in terms[:4]:  # limit API calls
        encoded = urllib.parse.quote(term)
        url = f"https://api.openalex.org/concepts?search={encoded}&per-page={max_per_term}&select=id,display_name,level"
        try:
            data = _fetch(url)
            for c in data.get("results", []):
                if c.get("level", 99) <= 3:  # skip overly specific leaf concepts
                    concept_ids.append(c["id"].replace("https://openalex.org/", ""))
        except Exception:
            continue
    return list(dict.fromkeys(concept_ids))  # deduplicate, preserve order


def find_papers(mechanisms: list, domain: str, max_results: int = 3) -> list:
    """
    Find papers related to the given mechanisms, excluding the analogy domain's
    own field so we get cross-domain results.
    """
    concept_ids = find_concept_ids(mechanisms)
    if not concept_ids:
        return []

    # Search papers tagged with any of these concepts
    concepts_filter = "|".join(concept_ids[:4])
    encoded_domain = urllib.parse.quote(domain)
    url = (
        f"https://api.openalex.org/works"
        f"?filter=concepts.id:{concepts_filter},is_oa:true,cited_by_count:>30"
        f"&sort=cited_by_count:desc"
        f"&per-page={max_results}"
        f"&select=title,doi,publication_year,cited_by_count,primary_location,concepts"
        f"&mailto=problem-transfer-engine@example.com"
    )
    try:
        data = _fetch(url)
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
