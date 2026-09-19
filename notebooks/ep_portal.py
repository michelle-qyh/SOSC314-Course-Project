"""Week 4: recover Parliament documents that CELLAR catalogues but does not hold.

49 in-scope works carry only a ``URI`` manifestation: the metadata record is held
centrally while the file stays on the issuing institution's own register. Council
documents are recoverable from the Council register (see collect.py). Parliament's
website returns an automated-access challenge, which we do not circumvent — but the
Parliament Open Data Portal exposes the same adopted texts through a public API.

Matching CELLAR works to Portal records is the non-obvious part. Title similarity
fails: CELLAR stores the long formal title ("European Parliament legislative
resolution of 13 March 2024 on the proposal for a regulation …") while the Portal
stores a short label ("Artificial Intelligence Act"), giving similarity scores
around 0.02–0.05 and confidently wrong matches. Date alone is ambiguous, because
Parliament adopts dozens of texts per sitting day. Date *and* EuroVoc AI subject tag
together are decisive, and a match is accepted only where exactly one AI-tagged
adopted text shares the document's date — an ambiguous day yields no match rather
than a guess.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

EP_API = "https://data.europarl.europa.eu/api/v2"
EUROVOC_AI_SUFFIX = "/3030"


def adopted_texts(years: list[int], pause: float = 0.4, timeout: int = 90) -> pd.DataFrame:
    """List Parliament adopted texts for the given years.

    Returns one row per text with its identifier, date, English title and whether
    it carries the EuroVoc concept for artificial intelligence.
    """
    rows = []
    for year in years:
        r = requests.get(f"{EP_API}/adopted-texts?year={year}&limit=900",
                         headers={"Accept": "application/ld+json"}, timeout=timeout)
        if r.status_code != 200:
            continue
        for item in r.json().get("data", []):
            title = item.get("title_dcterms", {})
            rows.append({
                "ep_id": item.get("identifier"),
                "date": item.get("document_date"),
                "title_en": title.get("en", "") if isinstance(title, dict) else str(title),
                "ai_tagged": any(EUROVOC_AI_SUFFIX in str(c) for c in item.get("is_about", [])),
            })
        time.sleep(pause)
    return pd.DataFrame(rows, columns=["ep_id", "date", "title_en", "ai_tagged"])


def match(targets: pd.DataFrame, listing: pd.DataFrame) -> pd.DataFrame:
    """Match unretrieved works to Portal records on date plus AI subject tag."""
    if not len(listing):
        return pd.DataFrame(columns=["doc_id", "date", "n_candidates", "ep_id"])
    out = []
    for _, row in targets.iterrows():
        key = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
        cand = listing[(listing["date"] == key) & listing["ai_tagged"]]
        out.append({"doc_id": row["doc_id"], "date": key, "n_candidates": len(cand),
                    "ep_id": cand.iloc[0]["ep_id"] if len(cand) == 1 else None})
    return pd.DataFrame(out)


def collect(targets: pd.DataFrame, raw_dir: Path, pause: float = 0.5,
            timeout: int = 90) -> pd.DataFrame:
    """Retrieve matched adopted texts, checkpointed like the CELLAR collector."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    log = []
    for _, row in targets.iterrows():
        path = raw_dir / f"{row.doc_id}.xhtml"
        if path.exists():
            log.append({"doc_id": row.doc_id, "status": "cached", "bytes": path.stat().st_size})
            continue
        try:
            r = requests.get(f"{EP_API}/adopted-texts/{row.ep_id}",
                             headers={"Accept": "text/html"}, timeout=timeout)
            r.raise_for_status()
            path.write_bytes(r.content)
            log.append({"doc_id": row.doc_id, "status": "ok", "bytes": len(r.content)})
        except Exception as exc:
            log.append({"doc_id": row.doc_id, "status": f"error: {type(exc).__name__}", "bytes": 0})
        time.sleep(pause)
    return pd.DataFrame(log, columns=["doc_id", "status", "bytes"])
