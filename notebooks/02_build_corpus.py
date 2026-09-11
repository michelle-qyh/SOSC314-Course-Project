# %% [markdown]
# # 02 — Corpus construction
#
# Takes the Week 2 inventory and produces the analysis corpus:
#
#     works_union.csv  ->  resolve formats  ->  retrieve  ->  extract
#                      ->  clean + segment  ->  relevance screen
#                      ->  documents.parquet / paragraphs.parquet
#
# Retrieval is checkpointed in `data/raw/`, so re-running skips files already on
# disk. A full cold run takes roughly three minutes; a warm run is instant.
#
# Run from the repository root:  `python notebooks/02_build_corpus.py`

# %%
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT / "notebooks"))

import build_corpus  # noqa: E402
import cellar  # noqa: E402
import collect  # noqa: E402

RAW = ROOT / "data" / "raw"
CORPUS = ROOT / "data" / "corpus"
CORPUS.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. In-scope documents
# Rebuild the in-scope set from the committed inventory and the current scope
# rules, so a change to `doc_types.csv` propagates without re-querying CELLAR.

# %%
works = pd.read_csv(ROOT / "data" / "inventory" / "works_union.csv")
doc_types = pd.read_csv(ROOT / "data" / "doc_types.csv")

joint = works["reason"].eq("jointly authored by multiple institutions")
works = works.drop(columns=["doc_type", "genre", "scope", "reason"]).merge(
    doc_types, on="type_raw", how="left")
works.loc[joint.values, "scope"] = "out"

core = works[works["scope"] == "in"].copy()
core["doc_id"] = core["work"].str.split("/").str[-1]
print(f"in scope: {len(core)}")
print(core.groupby("institution").size().to_string())

# %% [markdown]
# ## 2. Resolve the best available format per document
# CELLAR serves bytes at the item level, so we ask which manifestations exist
# for the English expression and take the most text-friendly one.

# %%
manifest = collect.manifestations(list(core["work"]))
best = collect.best_manifestation(manifest)
targets = core.merge(best, on="work", how="left")
targets[["doc_id", "work", "institution", "type_raw", "mtype"]].to_csv(
    CORPUS / "format_availability.csv", index=False)

print("\nformat resolved:")
print(targets["mtype"].value_counts(dropna=False).to_string())

# %% [markdown]
# ## 3. Retrieve
# Documents whose only manifestation is a `URI` pointer are hosted on the
# issuing institution's own register rather than in CELLAR. Council documents
# are recovered from the Council register; Parliament's are not reachable
# programmatically and are logged as unretrieved.

# %%
log = collect.collect(targets[targets["mtype"].notna()], RAW)
print(log["status"].value_counts().to_string())

missing = targets[targets["mtype"].isna()].copy()
refs = collect.council_references(missing["work"].tolist())
missing = missing.merge(refs, on="work", how="left")
council_log = collect.collect_council(missing[missing["ref"].notna()], RAW)
if len(council_log):
    print(council_log["status"].value_counts().to_string())

pd.concat([log, council_log]).to_csv(CORPUS / "retrieval_log.csv", index=False)

# %% [markdown]
# ## 4. Extract plain text

# %%
rows = []
for path in sorted(RAW.glob("*")):
    if path.name.startswith("."):
        continue
    rows.append({"doc_id": path.stem, "fmt": path.suffix.lstrip("."),
                 "text": collect.extract(path)})
extracted = pd.DataFrame(rows)
extracted["words"] = extracted["text"].str.split().map(len)
print(f"\nextracted: {len(extracted)}")
print(extracted.groupby("fmt")["words"].agg(["count", "median"]).round(0).to_string())

# %% [markdown]
# ## 5. Clean, segment, screen

# %%
meta = targets.drop(columns=["m"], errors="ignore")
docs, paras, exclusions = build_corpus.build(extracted, meta)

docs.to_parquet(CORPUS / "documents.parquet", index=False)
paras.to_parquet(CORPUS / "paragraphs.parquet", index=False)
exclusions.to_csv(CORPUS / "exclusions.csv", index=False)

# %% [markdown]
# ## 6. Descriptive summary

# %%
print(f"\nanalysis corpus: {len(docs)} documents | {len(paras):,} paragraphs "
      f"| {int(docs.n_words.sum()):,} tokens")
print("\nby institution:")
summary = docs.groupby("institution").agg(
    documents=("doc_id", "count"),
    median_words=("n_words", "median"),
    median_paragraphs=("n_paragraphs", "median"),
    median_ai_density=("ai_density", "median"),
).round(1)
print(summary.to_string())
print(f"\nmedian paragraph length: {int(paras.n_words.median())} words")
print(f"\nexcluded at screening: {len(exclusions)}")
print(exclusions["excluded_reason"].value_counts().to_string())
print("\nparagraphs containing an explicit AI reference (%):")
print((100 * (paras.ai_mentions > 0).groupby(paras.institution).mean()).round(0).to_string())
