# %% [markdown]
# # 01 — Corpus inventory and data feasibility
#
# Queries the CELLAR SPARQL endpoint for AI-related works authored by the
# European Commission, European Parliament and Council of the EU (2020–present),
# applies the scope rules documented in `data/doc_types.csv`, and produces the
# inventory figure used in `reports/01_data_feasibility.md`.
#
# Two retrieval criteria are combined:
# 1. **EuroVoc tag** — works indexed with the EuroVoc concept *artificial intelligence* (3030).
# 2. **Title match** — works whose English title contains "artificial intelligence".
#
# Run from the repository root: `python notebooks/01_inventory.py`
# (or open the paired `.ipynb`).

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT / "notebooks"))
import cellar  # noqa: E402

INV = ROOT / "data" / "inventory"
INV.mkdir(parents=True, exist_ok=True)
REFRESH = False  # set True to re-query CELLAR; False reuses cached CSVs

# %%
if REFRESH or not (INV / "works_eurovoc.csv").exists():
    ev = cellar.works_by_eurovoc(since="2020-01-01")
    ti = cellar.works_by_title(since="2020-01-01")
    for df, name in [(ev, "eurovoc"), (ti, "title")]:
        df["institution"] = df["agent"].map(cellar.institution)
        df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
        df.to_csv(INV / f"works_{name}.csv", index=False)
ev = pd.read_csv(INV / "works_eurovoc.csv").assign(hit="eurovoc")
ti = pd.read_csv(INV / "works_title.csv").assign(hit="title")

# %% [markdown]
# ## One row per work, restricted to the three institutions
# A work can carry several authoring bodies (e.g. COM and a DG). We keep the
# first institution match; works by other bodies (EESC, agencies, national
# parliaments) are dropped here and logged as out of scope.

# %%
u = pd.concat([ev, ti]).dropna(subset=["institution"])
u["type"] = u["type"].str.replace("resource-type/", "", regex=False)
works = (
    u.groupby("work")
    .agg(
        institution=("institution", "first"),
        type=("type", "first"),
        date=("date", "first"),
        title=("title", "first"),
        hits=("hit", lambda s: "+".join(sorted(set(s)))),
    )
    .reset_index()
)
works["year"] = pd.to_datetime(works["date"]).dt.year

# Works attributed to more than one institution (co-signed legal acts, joint
# declarations) cannot serve an institutional comparison and are excluded.
n_inst = u.drop_duplicates(["work", "institution"]).groupby("work")["institution"].nunique()
works["joint"] = works["work"].map(n_inst).gt(1)

# %% [markdown]
# ## Scope rules
# `data/doc_types.csv` maps every raw CELLAR resource type to a harmonised
# document type, a genre, and an in/out-of-scope decision with a reason.

# %%
doc_types = pd.read_csv(ROOT / "data" / "doc_types.csv")
works = works.merge(doc_types, left_on="type", right_on="type_raw", how="left")
unmapped = works[works["scope"].isna()]["type"].unique()
assert len(unmapped) == 0, f"Unmapped resource types: {unmapped}"
works.loc[works["joint"], ["scope", "reason"]] = ["out", "jointly authored by multiple institutions"]
works.to_csv(INV / "works_union.csv", index=False)

core = works[works["scope"] == "in"]
print("Total works found:", len(works))
print("In scope:", len(core))
print(core.groupby("institution").size())
print(core.pivot_table(index="year", columns="institution", values="work", aggfunc="count", fill_value=0))

# %% [markdown]
# ## Figure 1 — corpus inventory

# %%
INST = ["Commission", "Parliament", "Council"]
COLORS = {"Commission": "#1f4e79", "Parliament": "#c0504d", "Council": "#7f7f7f"}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})

# (a) in-scope documents by year and institution
byyear = core.pivot_table(index="year", columns="institution", values="work", aggfunc="count", fill_value=0)
byyear = byyear.reindex(columns=INST, fill_value=0)
byyear.plot(kind="bar", ax=axes[0], color=[COLORS[i] for i in INST], width=0.8, edgecolor="none")
axes[0].set_title("(a) In-scope AI documents in CELLAR, by year", loc="left", fontsize=10.5)
axes[0].set_xlabel("")
axes[0].set_ylabel("documents")
axes[0].tick_params(axis="x", rotation=0)
axes[0].legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.0, 0.83))
ymax = axes[0].get_ylim()[1]
axes[0].axvline(0.5, color="k", lw=0.6, ls=":")
axes[0].axvline(4.5, color="k", lw=0.6, ls=":")
axes[0].text(0.55, ymax * 0.97, "AI Act proposed\nApr 2021", fontsize=7.5, va="top")
axes[0].text(4.45, ymax * 0.97, "Digital Omnibus\non AI proposed\nNov 2025", fontsize=7.5, va="top", ha="right")
labels = [t.get_text() for t in axes[0].get_xticklabels()]
labels[-1] = labels[-1] + "*"
axes[0].set_xticklabels(labels)
axes[0].text(0.0, -0.16, "* 2026 covers January–August only", transform=axes[0].transAxes, fontsize=7.5)
axes[0].spines[["top", "right"]].set_visible(False)

# (b) composition by harmonised document type
comp = core.pivot_table(index="institution", columns="doc_type", values="work", aggfunc="count", fill_value=0)
comp = comp.reindex(INST)
order = comp.sum().sort_values(ascending=False).index
comp = comp[order]
comp.plot(kind="barh", stacked=True, ax=axes[1], colormap="tab20", edgecolor="white", linewidth=0.5)
axes[1].set_title("(b) Composition by harmonised document type", loc="left", fontsize=10.5)
axes[1].set_ylabel("")
axes[1].set_xlabel("documents")
axes[1].invert_yaxis()
axes[1].legend(frameon=False, fontsize=7.5, ncol=1, loc="center left", bbox_to_anchor=(1.0, 0.5))
axes[1].spines[["top", "right"]].set_visible(False)

fig.suptitle(
    f"Figure 1. Inventory of AI-related institutional documents retrievable from CELLAR "
    f"(n = {len(core)} in scope of {len(works)} found; 2020–{core['year'].max()})",
    fontsize=9.5, x=0.01, ha="left", y=1.02,
)
fig.tight_layout()
fig.savefig(ROOT / "reports" / "week2_corpus_inventory.png", dpi=200, bbox_inches="tight")
print("figure written")

# %% [markdown]
# ## Example documents (one per institution, in scope)

# %%
pd.set_option("display.max_colwidth", 120)
for inst in INST:
    print(f"\n### {inst}")
    print(core[core.institution == inst].sort_values("date")[["date", "doc_type", "title"]].head(5).to_string(index=False))
