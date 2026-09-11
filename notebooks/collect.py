"""Week 3: retrieve full texts for the in-scope corpus and extract plain text.

CELLAR serves a document's bytes at the *item* level, not the work level:

    work  ->  expression (one per language)  ->  manifestation (one per format)
                                                    ->  item  ({manifestation}/DOC_1)

Content negotiation on the work URI works for some documents only, so we resolve
manifestations explicitly via SPARQL and fetch the item URI directly. Formats are
tried in order of how cleanly they yield running text:

    xhtml  >  docx  >  pdf

Formex (fmx4) is *not* used: in this corpus its DOC_1 item is a bibliographic
wrapper whose body lives in a separately referenced physical file, so it yields
a few dozen words rather than the document. Every Formex-carrying work here also
offers XHTML, which is taken instead.

Retrieval is checkpointed: each document is written to data/raw/ once and skipped
on re-runs, so an interrupted collection resumes where it stopped.
"""
from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

import pandas as pd
import requests

import cellar

ENG = "http://publications.europa.eu/resource/authority/language/ENG"
FORMAT_PREFERENCE = ["xhtml", "docx", "pdf", "pdfa2a", "pdfx4"]
EXTENSION = {"xhtml": "xhtml", "docx": "docx",
             "pdf": "pdf", "pdfa2a": "pdf", "pdfx4": "pdf"}


# --------------------------------------------------------------------------
# 1. Discover which formats exist for each work
# --------------------------------------------------------------------------

def manifestations(work_uris: list[str], chunk: int = 30, pause: float = 0.5) -> pd.DataFrame:
    """One row per (work, manifestation, format) for the English expression."""
    frames = []
    for i in range(0, len(work_uris), chunk):
        vals = " ".join(f"<{w}>" for w in work_uris[i:i + chunk])
        q = f"""
        SELECT ?work ?m ?mtype WHERE {{
          VALUES ?work {{ {vals} }}
          ?e cdm:expression_belongs_to_work ?work ;
             cdm:expression_uses_language <{ENG}> .
          ?m cdm:manifestation_manifests_expression ?e ;
             cdm:manifestation_type ?mtype .
        }}"""
        frames.append(cellar.sparql(q))
        time.sleep(pause)
    return pd.concat(frames).drop_duplicates()


def best_manifestation(man: pd.DataFrame) -> pd.DataFrame:
    """Pick one manifestation per work, preferring structured formats over PDF."""
    rank = {f: i for i, f in enumerate(FORMAT_PREFERENCE)}
    m = man[man["mtype"].isin(rank)].copy()
    m["rank"] = m["mtype"].map(rank)
    return (m.sort_values(["work", "rank"])
              .groupby("work", as_index=False)
              .first()[["work", "m", "mtype"]])


# --------------------------------------------------------------------------
# 2. Retrieve the bytes
# --------------------------------------------------------------------------

def fetch_item(manifestation_uri: str, timeout: int = 120, max_item: int = 4) -> bytes:
    """Fetch a manifestation's document item.

    Items are usually numbered from DOC_1, but some manifestations start at
    DOC_2 (DOC_1 holding a cover or annex slot that was never populated), so we
    try successive item numbers until one resolves.
    """
    last = None
    for n in range(1, max_item + 1):
        r = requests.get(f"{manifestation_uri}/DOC_{n}", timeout=timeout)
        if r.status_code == 200 and len(r.content) > 2000:
            return r.content
        last = r
    last.raise_for_status()
    raise RuntimeError(f"no item found for {manifestation_uri}")


COUNCIL_PDF = "https://data.consilium.europa.eu/doc/document/{ref}/en/pdf"


def fetch_council_register(ref: str, timeout: int = 120) -> bytes:
    """Fetch a Council document from the Council's public register.

    49 in-scope works carry only a ``URI`` manifestation in CELLAR: the metadata
    record is held centrally but the file itself stays on the issuing
    institution's register. For the Council these are reachable at a stable
    public endpoint keyed by document reference (e.g. ST-8689-2026-INIT).
    """
    r = requests.get(COUNCIL_PDF.format(ref=ref), timeout=timeout,
                     headers={"User-Agent": "SOSC314 academic research (Duke Kunshan University)"})
    r.raise_for_status()
    return r.content


def collect(targets: pd.DataFrame, raw_dir: Path, pause: float = 0.4) -> pd.DataFrame:
    """Download every target that is not already on disk. Returns a retrieval log."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    log = []
    for _, row in targets.iterrows():
        path = raw_dir / f"{row.doc_id}.{EXTENSION[row.mtype]}"
        if path.exists():
            log.append({"doc_id": row.doc_id, "status": "cached", "bytes": path.stat().st_size})
            continue
        try:
            data = fetch_item(row.m)
            path.write_bytes(data)
            log.append({"doc_id": row.doc_id, "status": "ok", "bytes": len(data)})
        except Exception as exc:
            log.append({"doc_id": row.doc_id, "status": f"error: {type(exc).__name__}", "bytes": 0})
        time.sleep(pause)
    return pd.DataFrame(log)


def council_references(work_uris: list[str], chunk: int = 25, pause: float = 0.4) -> pd.DataFrame:
    """Extract Council register references (e.g. ST-8689-2026-INIT) for works
    whose only manifestation is a URI pointer."""
    frames = []
    for i in range(0, len(work_uris), chunk):
        vals = " ".join(f"<{w}>" for w in work_uris[i:i + chunk])
        q = f"""
        SELECT ?work ?o WHERE {{
          VALUES ?work {{ {vals} }}
          ?e cdm:expression_belongs_to_work ?work ;
             cdm:expression_uses_language <{ENG}> .
          ?m cdm:manifestation_manifests_expression ?e ;
             <http://www.w3.org/2002/07/owl#sameAs> ?o .
        }}"""
        frames.append(cellar.sparql(q))
        time.sleep(pause)
    if not frames:
        return pd.DataFrame(columns=["work", "ref"])
    sa = pd.concat(frames)
    sa["ref"] = sa["o"].str.extract(r"/consil/(ST_\d+_\d+_[A-Z]+)\.")
    out = sa.dropna(subset=["ref"]).drop_duplicates("work")[["work", "ref"]].copy()
    out["ref"] = out["ref"].str.replace("_", "-", regex=False)
    return out


def collect_council(targets: pd.DataFrame, raw_dir: Path, pause: float = 0.5) -> pd.DataFrame:
    """Retrieve Council-register documents not held in CELLAR."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    log = []
    for _, row in targets.iterrows():
        path = raw_dir / f"{row.doc_id}.pdf"
        if path.exists():
            log.append({"doc_id": row.doc_id, "status": "cached", "bytes": path.stat().st_size})
            continue
        try:
            data = fetch_council_register(row.ref)
            path.write_bytes(data)
            log.append({"doc_id": row.doc_id, "status": "ok", "bytes": len(data)})
        except Exception as exc:
            log.append({"doc_id": row.doc_id, "status": f"error: {type(exc).__name__}", "bytes": 0})
        time.sleep(pause)
    return pd.DataFrame(log)


# --------------------------------------------------------------------------
# 3. Extract plain text
# --------------------------------------------------------------------------

TAG = re.compile(r"<[^>]+>")
WS = re.compile(r"[ \t\xa0]+")
BLANKS = re.compile(r"\n{3,}")


def _from_markup(raw: bytes) -> str:
    """Strip tags from XHTML or Formex, keeping block boundaries as newlines."""
    text = raw.decode("utf-8", errors="replace")
    text = re.sub(r"(?is)<(script|style|head)[^>]*>.*?</\1>", " ", text)
    text = re.sub(r"(?i)</(p|div|tr|li|h[1-6]|P|TXT|TI|STI|ALINEA|NP)>", "\n", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = TAG.sub(" ", text)
    for entity, char in [("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"),
                         ("&gt;", ">"), ("&quot;", '"'), ("&#8217;", "'")]:
        text = text.replace(entity, char)
    return text


def _from_pdf(path: Path) -> str:
    """Extract text from a born-digital PDF via pdftotext."""
    out = subprocess.run(["pdftotext", "-nopgbrk", str(path), "-"],
                         capture_output=True, text=True, timeout=180)
    return out.stdout


def _from_docx(path: Path) -> str:
    """Extract paragraph text from a .docx by reading its XML."""
    import zipfile
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", errors="replace")
    xml = re.sub(r"(?i)</w:p>", "\n", xml)
    return TAG.sub("", xml)


def extract(path: Path) -> str:
    """Dispatch on file extension and return normalised plain text."""
    suffix = path.suffix.lower()
    if suffix in {".xhtml", ".xml", ".html"}:
        text = _from_markup(path.read_bytes())
    elif suffix == ".pdf":
        text = _from_pdf(path)
    elif suffix == ".docx":
        text = _from_docx(path)
    else:
        raise ValueError(f"no extractor for {suffix}")
    text = WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return BLANKS.sub("\n\n", text).strip()
