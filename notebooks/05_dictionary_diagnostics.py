# %% [markdown]
# # 05 — Diagnostics
#
# Four checks on the Week 4 measures:
#
#   1. term influence      does any single dictionary term carry a result?
#   2. term overlap        are the seven frames independent?
#   3. sampling            are the differences larger than the corpus can support?
#   4. preprocessing       which representation decision actually moves the answer?
#
# Run from the repository root:  `python notebooks/05_diagnostics.py`

# %%
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT / "notebooks"))

import diagnostics  # noqa: E402

ANALYSIS = ROOT / "data" / "analysis"
ANALYSIS.mkdir(parents=True, exist_ok=True)

paragraphs = pd.read_parquet(ROOT / "data" / "corpus" / "paragraphs.parquet")
print(f"corpus: {paragraphs.doc_id.nunique()} documents | {len(paragraphs):,} paragraphs")

hits, lengths = diagnostics.term_hits(paragraphs)
base = diagnostics.profile(paragraphs, hits, lengths)

# %% [markdown]
# ## 1. Term influence
# Refit the profile without each of the 152 dictionary terms in turn.

# %%
loo = diagnostics.leave_one_term_out(paragraphs, hits, lengths)
loo.to_csv(ANALYSIS / "leave_one_term_out.csv", index=False)
print("\nlargest single-term influence:")
print(loo.head(10).to_string(index=False))
print(f"\nterms moving a frame share by more than 1pp: {(loo.max_shift_pp > 1).sum()} of {len(loo)}")

tight = diagnostics.tightened_profile(paragraphs, hits, lengths)
comparison = pd.concat({"reported": base.round(1), "tightened": tight.round(1)}, axis=1)
comparison.to_csv(ANALYSIS / "frame_profile_tightened.csv")
print("\nprofile with 11 generic terms removed:")
print(comparison.to_string())
print(f"\nmax shift: {(base - tight).abs().max().max():.1f} pp")
print("frame rank order preserved:",
      bool((base.rank(ascending=False) == tight.rank(ascending=False)).all().all()))

# %% [markdown]
# ## 2. Term overlap
# Week 4 reported the frames as overlapping. How far do the lists actually share terms?

# %%
overlap = diagnostics.frame_overlap()
overlap.to_csv(ANALYSIS / "frame_overlap.csv", index=False)
print("\nterms appearing in more than one frame:")
print(overlap.to_string(index=False) if len(overlap) else "  none")

# %% [markdown]
# ## 3. Sampling
# Resample documents with replacement within each institution.

# %%
boot = diagnostics.bootstrap_profile(paragraphs, hits, lengths)
boot.to_csv(ANALYSIS / "bootstrap_frame_profile.csv", index=False)
print("\n95% confidence intervals:")
for frame in diagnostics.FRAMES:
    row = boot[boot.frame == frame].set_index("institution")
    cells = " | ".join(f"{i[:4]} {row.loc[i,'point']:5.1f} [{row.loc[i,'ci_low']:4.1f},{row.loc[i,'ci_high']:5.1f}]"
                       for i in ["Commission", "Parliament", "Council"])
    print(f"  {frame:26s} {cells}")

# %% [markdown]
# ## 4. Preprocessing, disentangled
# Each change measured against the right reference rather than all against baseline.

# %%
sens = diagnostics.sensitivity_clean(paragraphs)
sens.to_csv(ANALYSIS / "sensitivity_disentangled.csv", index=False)
print("\ntop-20 term overlap, each change against its own reference:")
print(sens.to_string(index=False))
