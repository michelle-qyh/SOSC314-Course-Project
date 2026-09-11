"""Week 3: turn retrieved files into the analysis corpus.

Input   data/raw/<doc_id>.<ext>      files retrieved by collect.py
Output  data/corpus/documents.parquet    one row per document
        data/corpus/paragraphs.parquet   one row per paragraph
        data/corpus/exclusions.csv       every dropped document, with a reason

Three preparation decisions are implemented here, each recorded per document so
the effect of any of them can be inspected or reversed:

1. Boilerplate removal — institutional documents carry recurring non-substantive
   material (publication headers, EEA-relevance notices, interinstitutional file
   numbers, footer pagination). These are matched by pattern and dropped.
2. Paragraph segmentation — frames are mixed within long documents, so the
   paragraph is the measurement unit. Segments shorter than MIN_PARA_WORDS are
   discarded as fragments (headings, list markers, table cells).
3. AI-relevance screen — retrieval selected documents by subject tag or title,
   which admits programme-level documents that mention AI in passing. Relevance
   is measured as mention *density* (AI terms per 1,000 words) rather than a raw
   count, so that a short procedural note naming AI twice is not penalised
   against a long programme evaluation naming it four times. Documents below
   MIN_DENSITY, or with fewer than MIN_AI_MENTIONS matches in total, leave the
   analysis corpus.
"""
from __future__ import annotations

import re

import pandas as pd

MIN_PARA_WORDS = 15
MIN_DOC_WORDS = 200
MIN_DENSITY = 0.5          # AI mentions per 1,000 words
MIN_AI_MENTIONS = 2          # absolute floor, guards against a single stray match

# Terms counted as an explicit reference to the subject. Deliberately narrow:
# generic technology words ("digital", "algorithm", "data") are excluded because
# they occur throughout EU digital-policy text regardless of whether AI is the
# subject, which would defeat the purpose of the screen.
AI_TERMS = re.compile(
    r"\b(artificial intelligence|\bAI\b|AI Act|machine learning|"
    r"general[- ]purpose AI|GPAI|foundation model|generative AI|"
    r"AI system|AI systems|algorithmic system)\b", re.IGNORECASE)

BOILERPLATE = [
    re.compile(r"(?im)^\s*(EN|FR|DE)\s*$"),
    re.compile(r"(?i)text with EEA relevance"),
    re.compile(r"(?im)^\s*Interinstitutional File:.*$"),
    re.compile(r"(?im)^\s*Official Journal of the European Union.*$"),
    re.compile(r"(?im)^\s*\d+\s*/\s*\d+\s*$"),
    re.compile(r"(?im)^\s*Brussels,\s+\d.*\(OR\..*\)\s*$"),
    re.compile(r"(?im)^\s*(LIMITE|RESTREINT UE/EU RESTRICTED)\s*$"),
    re.compile(r"(?im)^\s*www\.(europarl|consilium|europa)\.eu.*$"),
    re.compile(r"(?im)^\s*Publications Office of the European Union.*$"),
    re.compile(r"(?im)^\s*ISBN[\s:0-9-]+$"),
    re.compile(r"(?im)^\s*doi:.*$"),
    # Council routing headers and dossier subject-code strings, e.g.
    # "JUSTCIV 149 JAI 1313 CONSOM 364 COMPET 989 CODEC 1882"
    re.compile(r"(?m)^(?:\s*[A-Z][A-Z.]{2,14}\s+\d{1,5}\b){2,}\s*$"),
    re.compile(r"(?im)^\s*(From|To|Subject|No\. prev\. doc\.|No\. Cion doc\.)\s*:\s*.*$"),
    re.compile(r"(?im)^\s*(OUTCOME OF PROCEEDINGS|NOTE|COVER NOTE|I/A ITEM NOTE)\s*$"),
    re.compile(r"(?im)^\s*\d{4}/\d{4}\s*\(COD\)\s*$"),
    re.compile(r"(?im)^\s*(Delegations will find|Delegations will receive).*$"),
]

SPLIT = re.compile(r"\n\s*\n+")


def strip_boilerplate(text: str) -> str:
    for pattern in BOILERPLATE:
        text = pattern.sub(" ", text)
    return re.sub(r"\n{3,}", "\n\n", text)


def paragraphs(text: str) -> list[str]:
    """Split into paragraphs and drop fragments below the length floor."""
    out = []
    for block in SPLIT.split(text):
        block = " ".join(block.split())
        if len(block.split()) >= MIN_PARA_WORDS:
            out.append(block)
    return out


def ai_mentions(text: str) -> int:
    return len(AI_TERMS.findall(text))


def build(extracted: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean, segment and screen. Returns (documents, paragraphs, exclusions)."""
    df = meta.merge(extracted, on="doc_id", how="inner").copy()
    df["text_clean"] = df["text"].map(strip_boilerplate)
    df["paras"] = df["text_clean"].map(paragraphs)
    df["n_paragraphs"] = df["paras"].map(len)
    df["n_words"] = df["text_clean"].str.split().map(len)
    df["ai_mentions"] = df["text_clean"].map(ai_mentions)
    df["ai_density"] = 1000 * df["ai_mentions"] / df["n_words"].clip(lower=1)

    too_short = df["n_words"] < MIN_DOC_WORDS
    peripheral = (~too_short) & ((df["ai_density"] < MIN_DENSITY) |
                                 (df["ai_mentions"] < MIN_AI_MENTIONS))
    df["excluded_reason"] = ""
    df.loc[too_short, "excluded_reason"] = f"under {MIN_DOC_WORDS} words after cleaning"
    df.loc[peripheral, "excluded_reason"] = (
        f"AI peripheral (<{MIN_DENSITY} mentions per 1,000 words or <{MIN_AI_MENTIONS} mentions)")

    exclusions = df[df.excluded_reason != ""][
        ["doc_id", "institution", "doc_type", "n_words", "ai_mentions",
         "ai_density", "excluded_reason"]].copy()
    keep = df[df.excluded_reason == ""].copy()

    para_rows = []
    for _, r in keep.iterrows():
        for i, p in enumerate(r["paras"]):
            para_rows.append({"doc_id": r.doc_id, "para_idx": i, "institution": r.institution,
                              "doc_type": r.doc_type, "genre": r.genre, "year": r.year,
                              "n_words": len(p.split()), "ai_mentions": ai_mentions(p), "text": p})
    para = pd.DataFrame(para_rows)

    docs = keep.drop(columns=["paras", "text", "excluded_reason"])
    return docs, para, exclusions
