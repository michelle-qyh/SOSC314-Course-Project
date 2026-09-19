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
import ep_portal  # noqa: E402

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

# Parliament documents are not held in CELLAR, but adopted texts are available
# through the Parliament Open Data Portal. Matching is by date and AI subject tag:
# date alone is ambiguous (Parliament adopts dozens of texts per sitting day) and
# title similarity is unreliable, because CELLAR stores the long formal title while
# the Portal stores a short label. A match is accepted only where exactly one
# AI-tagged adopted text shares the date, so an ambiguous day yields no match
# rather than a guess.
ep_log = pd.DataFrame(columns=["doc_id", "status", "bytes"])
ep_missing = missing.merge(core[["doc_id", "type_raw"]], on="doc_id", how="left", suffixes=("", "_c"))
ep_missing = ep_missing[(ep_missing["ref"].isna())
                        & (ep_missing["institution"] == "Parliament")
                        & (ep_missing["type_raw"] == "ADOPT_TEXT")]
if len(ep_missing):
    # The Portal is a secondary source; if it is unreachable the pipeline continues
    # with the CELLAR-held documents rather than aborting.
    try:
        years = sorted(pd.to_datetime(ep_missing["date"]).dt.year.unique())
        listing = ep_portal.adopted_texts([int(y) for y in years])
        matched = ep_portal.match(ep_missing, listing)
        print(f"\nParliament Open Data Portal: {matched.ep_id.notna().sum()} of {len(matched)} "
              f"adopted texts uniquely identified")
        ep_log = ep_portal.collect(matched[matched.ep_id.notna()], RAW)
        if len(ep_log):
            print(ep_log["status"].value_counts().to_string())
    except Exception as exc:
        print(f"\nParliament Open Data Portal unavailable ({type(exc).__name__}); continuing without it")

# Documents with no CELLAR manifestation are recorded explicitly rather than
# omitted, so the log accounts for every in-scope document: a reader can see
# that these were unreachable, not overlooked.
# "Unreachable" is decided by what is actually on disk, not by this run's outcome:
# a document retrieved earlier is still retrieved, and a secondary source that
# happens to be down today must not turn a held document into a missing one.
on_disk = {q.stem for q in RAW.glob("*") if not q.name.startswith(".")}

late = missing[missing["ref"].isna() & missing["doc_id"].isin(on_disk)][["doc_id"]].copy()
if len(late):
    late["status"] = "ok (Parliament Open Data Portal)"
    late["bytes"] = [next((q.stat().st_size for q in RAW.glob(f"{d}.*")), 0) for d in late["doc_id"]]
    ep_log = pd.concat([ep_log, late], ignore_index=True).drop_duplicates("doc_id", keep="last")

unreachable = missing[missing["ref"].isna() & ~missing["doc_id"].isin(on_disk)][["doc_id"]].copy()
unreachable["status"] = "not retrievable: no CELLAR manifestation (held on the issuing institution's own register)"
unreachable["bytes"] = 0

retrieval_log = pd.concat([log, council_log, ep_log, unreachable], ignore_index=True)
retrieval_log = retrieval_log.merge(
    targets[["doc_id", "institution", "type_raw", "mtype"]], on="doc_id", how="left")
retrieval_log.to_csv(CORPUS / "retrieval_log.csv", index=False)
print(f"\nretrieval log: {len(retrieval_log)} rows covering all {len(core)} in-scope documents")
print(retrieval_log.groupby("institution")["status"].value_counts().to_string())

# %% [markdown]
# ## 4. Extract plain text

# %%
# One file per document. A document can leave more than one file in data/raw if an
# earlier run chose a different format, so we take a single file per doc_id in the
# order of FORMAT_PREFERENCE rather than globbing the directory, which would enter
# the same document into the corpus twice.
by_doc: dict[str, list] = {}
for path in sorted(RAW.glob("*")):
    if path.name.startswith("."):
        continue
    by_doc.setdefault(path.stem, []).append(path)

ext_rank = {ext: i for i, ext in enumerate(["xhtml", "xml", "html", "docx", "pdf"])}
rows = []
for doc_id, paths in by_doc.items():
    chosen = sorted(paths, key=lambda q: ext_rank.get(q.suffix.lstrip("."), 99))[0]
    rows.append({"doc_id": doc_id, "fmt": chosen.suffix.lstrip("."),
                 "text": collect.extract(chosen)})
extracted = pd.DataFrame(rows)
extracted["words"] = extracted["text"].str.split().map(len)
print(f"\nfiles on disk: {sum(len(v) for v in by_doc.values())} | documents extracted: {len(extracted)}")
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
