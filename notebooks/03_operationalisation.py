# %% [markdown]
# # 03 — Operationalisation and analytic comparison
#
# Two routes from the corpus to the project's hypotheses:
#
#   dictionary scoring   theory-driven; the seven frames of the proposal as term
#                        lists, scored per paragraph and aggregated to documents
#   Fightin' Words       data-driven; the vocabulary that distinguishes each pair
#                        of institutions, with no categories imposed
#
# Each is run under six representation specifications, so that the effect of the
# preprocessing choice is measured rather than assumed.
#
# Run from the repository root:  `python notebooks/03_operationalisation.py`

# %%
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT / "notebooks"))

import measures  # noqa: E402
import represent  # noqa: E402

CORPUS = ROOT / "data" / "corpus"
ANALYSIS = ROOT / "data" / "analysis"
ANALYSIS.mkdir(parents=True, exist_ok=True)

INSTITUTIONS = ["Commission", "Parliament", "Council"]
PAIRS = [("Commission", "Parliament"), ("Parliament", "Council"), ("Commission", "Council")]
FRAMES = list(measures.FRAMES)

paragraphs = pd.read_parquet(CORPUS / "paragraphs.parquet")
print(f"corpus: {paragraphs.doc_id.nunique()} documents | {len(paragraphs):,} paragraphs")

# %% [markdown]
# ## 1. Dictionary scoring
# Frame scores are matched terms per 100 tokens. Paragraph scores are averaged to
# the document, then across documents, so that a long document does not count for
# more than a short one. Profiles are reported as each institution's share of its
# own total frame signal, which removes the overall AI-density differences already
# described in Week 3 and leaves the relative emphasis.

# %%
def frame_profile(df: pd.DataFrame, **tokenise_kwargs) -> pd.DataFrame:
    scores = measures.score_frames(df["text"], **tokenise_kwargs)
    scores["doc_id"] = df["doc_id"].values
    scores["institution"] = df["institution"].values
    per_doc = scores.groupby(["doc_id", "institution"])[FRAMES].mean().reset_index()
    per_inst = per_doc.groupby("institution")[FRAMES].mean()
    return (per_inst.T / per_inst.T.sum() * 100)


profile = frame_profile(paragraphs)
profile.round(1).to_csv(ANALYSIS / "frame_profile.csv")
print("\nframe profile (% of each institution's total frame signal):")
print(profile.round(1).to_string())

# %% [markdown]
# ## 2. Fightin' Words
# Log-odds with an informative Dirichlet prior, for each pair of institutions and
# each specification. The top terms are written out so the ranking can be read
# directly rather than summarised.

# %%
rows = []
for spec in represent.SPECIFICATIONS:
    for a, b in PAIRS:
        fw = measures.fightin_words(paragraphs.loc[paragraphs.institution == a, "text"],
                                    paragraphs.loc[paragraphs.institution == b, "text"], spec=spec)
        top_a, top_b = measures.top_terms(fw, 20)
        rows.append({"specification": spec, "group": a, "against": b, "terms": ", ".join(top_a)})
        rows.append({"specification": spec, "group": b, "against": a, "terms": ", ".join(top_b)})
distinctive = pd.DataFrame(rows)
distinctive.to_csv(ANALYSIS / "distinctive_terms.csv", index=False)

print("\ndistinctive terms, baseline vs house-style-removed (Parliament against Council):")
for spec in ("baseline", "drop_house_style"):
    sel = distinctive[(distinctive.specification == spec) & (distinctive.group == "Parliament")]
    print(f"  {spec:18s} {sel.iloc[0].terms[:95]}")

# %% [markdown]
# ## 3. Controlled comparison
# How far does each measure move when the representation changes? For Fightin'
# Words, the overlap of the top-20 terms with the baseline. For the dictionary, the
# shift in frame shares and whether the rank order of frames survives.

# %%
baseline_terms = {(r.group, r.against): set(r.terms.split(", "))
                  for _, r in distinctive[distinctive.specification == "baseline"].iterrows()}
stability = []
for spec in list(represent.SPECIFICATIONS)[1:]:
    overlaps = []
    for _, r in distinctive[distinctive.specification == spec].iterrows():
        base = baseline_terms[(r.group, r.against)]
        overlaps.append(len(base & set(r.terms.split(", "))) / 20)
    stability.append({"specification": spec, "mean_overlap": round(sum(overlaps) / len(overlaps), 2)})
stability = pd.DataFrame(stability)
stability.to_csv(ANALYSIS / "fw_stability.csv", index=False)
print("\nFightin' Words — overlap of top-20 terms with baseline:")
print(stability.to_string(index=False))

profile_alt = frame_profile(paragraphs, drop_procedural=True)
shift = (profile - profile_alt).abs()
print(f"\ndictionary — mean shift {shift.mean().mean():.2f} pp, max {shift.max().max():.2f} pp")
print("frame rank order per institution unchanged:",
      bool((profile.rank(ascending=False) == profile_alt.rank(ascending=False)).all().all()))

# %% [markdown]
# ## 4. Genre control
# Document type is confounded with institution. Restricting to the formal-position
# genre tests whether the profiles are a property of the institution or of the
# genres it happens to produce.

# %%
formal = paragraphs[paragraphs.genre == "formal_position"]
profile_formal = frame_profile(formal)
profile_formal.round(1).to_csv(ANALYSIS / "frame_profile_formal_position.csv")
print("\nformal-position genre only:")
print(profile_formal.round(1).to_string())
print("\ndocuments per institution in the genre subset:")
print(formal.groupby("institution").doc_id.nunique().to_string())
