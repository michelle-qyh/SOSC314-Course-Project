# %% [markdown]
# # 06 — Face validity and inherited text
#
# Two checks that the Week 5 diagnostics pointed to but could not perform, because
# both require reading the corpus rather than computing over it.
#
#   face validity     do the paragraphs a frame scores highest actually read as that
#                     frame? The diagnostics established that the dictionary does not
#                     depend on any single term and survives resampling, but nothing
#                     so far tested whether a high score means what it claims.
#
#   inherited text    Council compromise texts and Parliament amendment lists
#                     reproduce Commission base text. Flagged in Week 3, never
#                     measured. If an institution's paragraphs are largely another's
#                     words, a comparison between them is partly a comparison of one
#                     institution with itself.
#
# This is not validation against human coding: there is no second coder and no
# agreement statistic. It is the weaker check of reading the top-scoring paragraphs
# and reporting what they contain, with the paragraphs themselves written out so the
# judgement can be checked rather than taken on trust.
#
# Run from the repository root:  `python notebooks/06_face_validity.py`

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT / "notebooks"))

import diagnostics  # noqa: E402

ANALYSIS = ROOT / "data" / "analysis"
ANALYSIS.mkdir(parents=True, exist_ok=True)

FRAMES = diagnostics.FRAMES
MIN_TOKENS = 30   # short fragments score high on little evidence; excluded from the read
TOP_N = 10        # paragraphs read per frame
KEY_CHARS = 200   # prefix length used to detect near-identical paragraphs

paragraphs = pd.read_parquet(ROOT / "data" / "corpus" / "paragraphs.parquet")
hits, lengths = diagnostics.term_hits(paragraphs)

scores = pd.DataFrame({
    frame: 100 * sum(hits[frame].values()) / np.maximum(lengths, 1) for frame in FRAMES})
scores["n_tokens"] = lengths
for column in ["doc_id", "institution", "text"]:
    scores[column] = paragraphs[column].values
print(f"scored {len(scores):,} paragraphs")

# %% [markdown]
# ## 1. Face validity
# The ten highest-scoring paragraphs per frame, long enough to judge.

# %%
readable = scores[scores.n_tokens >= MIN_TOKENS]
print(f"paragraphs of at least {MIN_TOKENS} tokens: {len(readable):,}")


def normalise(series: pd.Series) -> pd.Series:
    """Lowercased, punctuation-free prefix used to spot near-identical paragraphs."""
    return (series.str.lower()
                  .str.replace(r"[^a-z ]", "", regex=True)
                  .str.replace(r"\s+", " ", regex=True)
                  .str[:KEY_CHARS])


records = []
for frame in FRAMES:
    top = readable.nlargest(TOP_N, frame)
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        records.append({"frame": frame, "rank": rank, "institution": row.institution,
                        "score": round(row[frame], 1), "n_tokens": int(row.n_tokens),
                        "doc_id": row.doc_id, "text": row.text[:500]})
pd.DataFrame(records).to_csv(ANALYSIS / "face_validity_top_paragraphs.csv", index=False)

summary = []
for frame in FRAMES:
    top = readable.nlargest(TOP_N, frame)
    summary.append({"frame": frame,
                    "top_score": round(top[frame].max(), 1),
                    "distinct_texts_in_top10": normalise(top.text).nunique()})
summary = pd.DataFrame(summary)
print("\nhighest-scoring paragraphs per frame:")
print(summary.to_string(index=False))
print(f"\nthe {TOP_N * len(FRAMES)} paragraphs are written to face_validity_top_paragraphs.csv;"
      "\nthe verdict column in face_validity_summary.csv records what reading them showed.")

# %% [markdown]
# ## 2. Inherited text
# Paragraphs whose opening is near-identical to a paragraph in another institution's
# documents. A high figure means the institution's corpus is partly another's words.

# %%
scores["key"] = normalise(scores.text)
institutions_per_key = scores.groupby("key")["institution"].nunique()
shared_keys = set(institutions_per_key[institutions_per_key > 1].index)

totals = scores.groupby("institution").size()
inherited = scores[scores.key.isin(shared_keys)].groupby("institution").size()
inherited_table = pd.DataFrame({
    "paragraphs": totals,
    "cross_institution_duplicates": inherited,
    "pct": (100 * inherited / totals).round(1)}).reset_index()
inherited_table.to_csv(ANALYSIS / "inherited_text_by_institution.csv", index=False)
print("\nparagraphs shared near-verbatim with another institution:")
print(inherited_table.to_string(index=False))

# %% [markdown]
# ## 3. Does inherited text change the result?

# %%
def frame_profile(df: pd.DataFrame) -> pd.DataFrame:
    per_doc = df.groupby(["doc_id", "institution"])[FRAMES].mean().reset_index()
    per_inst = per_doc.groupby("institution")[FRAMES].mean()
    return per_inst.T / per_inst.T.sum() * 100


with_all = frame_profile(scores)
without_inherited = frame_profile(scores[~scores.key.isin(shared_keys)])
pd.concat({"all_paragraphs": with_all.round(2),
           "inherited_removed": without_inherited.round(2)}, axis=1).to_csv(
    ANALYSIS / "frame_profile_inherited_removed.csv")

shift = (with_all - without_inherited).abs().max().max()
print(f"\nmax shift when inherited paragraphs are removed: {shift:.1f} pp")
print("frame rank order preserved:",
      bool((with_all.rank(ascending=False) ==
            without_inherited.rank(ascending=False)).all().all()))
