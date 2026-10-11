# %% [markdown]
# # 07 — Figures
#
# Redraws, from the committed data alone, the report and presentation figures that
# were not produced by an earlier script:
#
#   figure1_operationalisation_week4.png   frame profile and Fightin' Words sensitivity (Week 4)
#   figure1_diagnostics_week5.png          bootstrap intervals and the effect of each decision (Week 5)
#   figure3_face_validity_week5.png        top-scoring paragraphs for one passing, one failing frame (Week 5)
#   figure_corpus_by_year_final.png        analysis corpus by year and institution (presentation)
#   figure_corpus_attrition_final.png      in scope → retrieved → analysed, by institution (presentation)
#
# Inputs are the CSV and parquet files in data/; nothing is recomputed here, so the
# figures show exactly the numbers in the committed results. Run 03, 05 and 06 first
# if those files are missing.
#
# Typeface: EB Garamond if installed (as in the reports), otherwise the default serif.
#
# Run from the repository root:  `python notebooks/07_figures.py`

# %%
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT / "notebooks"))

import measures  # noqa: E402

DATA = ROOT / "data"
ANALYSIS = DATA / "analysis"
FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

FACE = "EB Garamond" if any(f.name == "EB Garamond" for f in font_manager.fontManager.ttflist) else "serif"
plt.rcParams.update({
    "font.family": [FACE, "DejaVu Serif"],   # DejaVu supplies glyphs such as → if the main face lacks them
    "font.size": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

INSTITUTIONS = ["Commission", "Parliament", "Council"]
COLOUR = {"Commission": "#1F4E79", "Parliament": "#C0504D", "Council": "#7F7F7F"}

# frames in the order used by every report figure, top to bottom
FRAME_ORDER = [
    ("technological_innovation", "Technological innovation"),
    ("economic_opportunity", "Economic opportunity"),
    ("consumer_protection", "Consumer protection"),
    ("regulatory_governance", "Regulatory governance"),
    ("fundamental_rights", "Fundamental rights"),
    ("labour_social", "Labour and social"),
    ("security_geopolitical", "Security / geopolitics"),
]
# vertical offset of each institution inside a frame group (Council drawn on top)
OFFSET = {"Council": 0.27, "Parliament": 0.0, "Commission": -0.27}


def save(fig, name):
    fig.savefig(FIGURES / name)
    plt.close(fig)
    print(f"saved reports/figures/{name}")


# %% [markdown]
# ## Week 4, Figure 1 — frame profile and Fightin' Words sensitivity

# %%
profile = pd.read_csv(ANALYSIS / "frame_profile.csv", index_col=0)
fw = pd.read_csv(ANALYSIS / "fw_stability.csv").set_index("specification")["mean_overlap"]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.9, 5.2), gridspec_kw={"width_ratios": [1.35, 1]})
y = np.arange(len(FRAME_ORDER))[::-1]
for inst in INSTITUTIONS:
    ax.barh(y + OFFSET[inst], [profile.loc[f, inst] for f, _ in FRAME_ORDER],
            height=0.27, color=COLOUR[inst], alpha=0.85, label=inst)
ax.set_yticks(y, [label for _, label in FRAME_ORDER])
ax.set_xlabel("share of the institution's total frame signal (%)")
ax.set_title("(a) Frame profile — dictionary scoring", loc="left")
handles, labels = ax.get_legend_handles_labels()
order = [labels.index(i) for i in INSTITUTIONS]
ax.legend([handles[i] for i in order], INSTITUTIONS, frameon=False, loc="lower right")

specs = [("drop_procedural", "drop procedural"), ("drop_house_style", "drop house style"),
         ("with_bigrams", "with bigrams"), ("tfidf", "tfidf"), ("min_df_20", "min df 20")]
yb = np.arange(len(specs))[::-1]
bx.barh(yb, [fw[s] for s, _ in specs], height=0.55, color="#A38B5F")
for yy, (s, _) in zip(yb, specs):
    bx.text(fw[s] + 0.02, yy, f"{fw[s]:.2f}", va="center", fontsize=11)
bx.set_yticks(yb, [label for _, label in specs])
bx.set_xlim(0, 1.25)
bx.set_ylim(-0.6, len(specs) + 0.3)
bx.axvline(1.0, color="#3B6E2F", ls="--", lw=2)
bx.annotate("dictionary: frame rank order\nunchanged under every variant", xy=(1.0, len(specs) - 0.45),
            xytext=(0.58, len(specs) - 0.2), color="#3B6E2F", style="italic", fontsize=10,
            arrowprops=dict(arrowstyle="->", color="#3B6E2F"))
bx.set_xlabel("overlap of top-20 distinctive terms with baseline")
bx.set_title("(b) Sensitivity to representation choices", loc="left")
fig.tight_layout()
save(fig, "figure1_operationalisation_week4.png")

# %% [markdown]
# ## Week 5, Figure 1 — bootstrap intervals and what moves the answer
# Panel (b) mixes two units on one axis, as in the report: Fightin' Words turnover
# (% of top-20 terms replaced) and the dictionary's largest frame-share shift (pp).

# %%
boot = pd.read_csv(ANALYSIS / "bootstrap_frame_profile.csv").set_index(["institution", "frame"])
sens = pd.read_csv(ANALYSIS / "sensitivity_disentangled.csv").set_index("change")["overlap"]
tight = pd.read_csv(ANALYSIS / "frame_profile_tightened.csv", header=[0, 1], index_col=0)
dictionary_shift = (tight["reported"] - tight["tightened"]).abs().to_numpy().max()

fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.3, 5.45), gridspec_kw={"width_ratios": [1.25, 1]})
for inst in INSTITUTIONS:
    rows = [boot.loc[(inst, f)] for f, _ in FRAME_ORDER]
    pts = np.array([r.point for r in rows])
    err = np.array([[r.point - r.ci_low for r in rows], [r.ci_high - r.point for r in rows]])
    ax.errorbar(pts, y + OFFSET[inst], xerr=err, fmt="o", color=COLOUR[inst], ms=6,
                capsize=3, elinewidth=1.5, label=inst)
ax.set_yticks(y, [label for _, label in FRAME_ORDER])
ax.set_xlim(-0.5, 41)
ax.grid(axis="x", color="#E6E6E6")
ax.set_axisbelow(True)
ax.set_xlabel("share of the institution's total frame signal (%), 95% CI")
ax.set_title("(a) Bootstrap confidence intervals", loc="left")
ax.legend(frameon=False, loc="lower right")

effects = [  # label, effect, colour (top to bottom)
    ("TF-IDF weighting", 100 * (1 - sens["tfidf"]), "#C9D6E3"),
    ("min_df 5 → 20", 100 * (1 - sens["min_df_20"]), "#C9D6E3"),
    ("add bigrams", 100 * (1 - sens["with_bigrams"]), "#6F94BF"),
    ("stopword lists\n(procedural + rhetorical)", 100 * (1 - sens["stopword lists"]), "#1F4E79"),
    ("dictionary terms\n(11 generic removed)", dictionary_shift, "#A23434"),
]
ye = np.arange(len(effects))[::-1]
bx.barh(ye, [e for _, e, _ in effects], height=0.55, color=[c for _, _, c in effects])
for yy, (_, e, _) in zip(ye, effects):
    bx.text(e + 0.6, yy, f"{e:.0f}", va="center", fontsize=11)
bx.set_yticks(ye, [label for label, _, _ in effects])
bx.set_xlim(0, 34)
bx.set_xlabel("effect on the result (% of top-20 terms replaced,\nor percentage-point shift in frame share)")
bx.set_title("(b) What actually moves the answer", loc="left")
fig.text(0.99, -0.02, "blue: Fightin' Words, top-20 term turnover  —  red: dictionary, frame-share shift",
         ha="right", style="italic", color="#777777", fontsize=10)
fig.tight_layout()
save(fig, "figure1_diagnostics_week5.png")

# %% [markdown]
# ## Week 5, Figure 3 — face validity
# The four highest-scoring paragraphs for one frame that passes and one that fails,
# with the dictionary's single-word terms highlighted.

# %%
top = pd.read_csv(ANALYSIS / "face_validity_top_paragraphs.csv")
PANELS = [
    ("fundamental_rights", "(a) Fundamental rights — passes", "#3B7A3B",
     "four different passages; the matched terms carry the rights content"),
    ("technological_innovation", "(b) Technological innovation — fails", "#A23434",
     "all four are one annex list, reproduced in Parliament, Commission and Council documents"),
]
HIGHLIGHT = "#B8650F"
MAX_CHARS = 145   # excerpt length before the ellipsis
N_SHOWN = 4


def excerpt(text):
    """Cut a paragraph at a word boundary near MAX_CHARS and mark the cut."""
    if len(text) <= MAX_CHARS:
        return text
    return text[:MAX_CHARS].rsplit(" ", 1)[0] + " ..."


def core(token):
    return "".join(ch for ch in token.lower() if ch.isalpha() or ch == "-").strip("-")


fig = plt.figure(figsize=(7.98, 8.39))
renderer = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()
LEFT, INDENT, RIGHT, SIZE, LINE = 0.015, 0.045, 0.985, 12, 0.026


def width(word):
    t = fig.text(0, 0, word, fontsize=SIZE)
    w = inv.transform((t.get_window_extent(renderer).width, 0))[0] - inv.transform((0, 0))[0]
    t.remove()
    return w


space = width("a b") - width("ab")
yy = 0.985
for frame, heading, colour, subtitle in PANELS:
    terms = {t for t in measures.FRAMES[frame] if " " not in t}
    fig.text(LEFT, yy, heading, fontsize=18, color=colour, va="top")
    yy -= 0.036
    fig.text(LEFT, yy, subtitle, fontsize=12, style="italic", color="#666666", va="top")
    yy -= 0.05
    rows = top[top.frame == frame].sort_values("rank").head(N_SHOWN)
    for _, row in rows.iterrows():
        fig.text(LEFT, yy, str(int(row["rank"])), fontsize=SIZE, color="#333333", va="top")
        fig.text(INDENT, yy, row["institution"], fontsize=SIZE, color=COLOUR[row["institution"]], va="top")
        fig.text(RIGHT, yy, f"score {row['score']:.1f}", fontsize=11, color="#666666", va="top", ha="right")
        yy -= LINE + 0.004
        x = INDENT
        for word in excerpt(str(row["text"])).split():
            w = width(word)
            if x + w > RIGHT:
                x, yy = INDENT, yy - LINE
            fig.text(x, yy, word, fontsize=SIZE, va="top",
                     color=HIGHLIGHT if core(word) in terms else "#222222")
            x += w + space
        yy -= LINE + 0.024
    yy -= 0.012
save(fig, "figure3_face_validity_week5.png")

# %% [markdown]
# ## Presentation — the corpus by year, and attrition by institution
# In scope = every document in the retrieval log; retrieved = text obtained from
# CELLAR, the Council register or the Parliament Open Data Portal; analysed = the
# final corpus after the relevance screen.

# %%
docs = pd.read_parquet(DATA / "corpus" / "documents.parquet")
log = pd.read_csv(DATA / "corpus" / "retrieval_log.csv")
excluded = pd.read_csv(DATA / "corpus" / "exclusions.csv")

by_year = pd.crosstab(docs["year"], docs["institution"])[INSTITUTIONS]
fig, ax = plt.subplots(figsize=(9, 5.2))
x = np.arange(len(by_year))
for i, inst in enumerate(INSTITUTIONS):
    ax.bar(x + (i - 1) * 0.27, by_year[inst], 0.27, color=COLOUR[inst], label=f"{inst} ({by_year[inst].sum()})")
ax.set_xticks(x, [f"{yr}*" if yr == by_year.index.max() else str(yr) for yr in by_year.index])
for year, label in [(2021, "AI Act proposed\nApr 2021"), (2025, "Digital Omnibus\non AI, Nov 2025")]:
    pos = list(by_year.index).index(year) - 0.5 + (1 if year == 2025 else 0)
    ax.axvline(pos, ls=":", color="black", lw=1)
    ax.text(pos + 0.05, by_year.to_numpy().max() * 0.97, label, fontsize=11, va="top")
ax.set_ylabel("documents in analysis corpus")
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.36, 0.99))
ax.text(0, -0.13, f"* {by_year.index.max()} = January–{pd.to_datetime(docs['date']).max():%B}",
        transform=ax.transAxes, fontsize=10, color="#555555")
save(fig, "figure_corpus_by_year_final.png")

retrieved_ids = set(log.loc[log.status == "ok", "doc_id"]) | set(docs.doc_id) | set(excluded.doc_id)
stages = pd.DataFrame({
    "In scope (scope rules)": log.groupby("institution").size(),
    "Full text retrieved": log[log.doc_id.isin(retrieved_ids)].groupby("institution").size(),
    "Analysis corpus": docs.groupby("institution").size(),
}).loc[INSTITUTIONS]
shade = {"In scope (scope rules)": 0.25, "Full text retrieved": 0.6, "Analysis corpus": 1.0}

fig, ax = plt.subplots(figsize=(9, 5.2))
x = np.arange(len(INSTITUTIONS))
for k, stage in enumerate(stages.columns):
    for j, inst in enumerate(INSTITUTIONS):
        n = stages.loc[inst, stage]
        ax.bar(x[j] + (k - 1) * 0.26, n, 0.26, color=COLOUR[inst], alpha=shade[stage], edgecolor="white")
        ax.text(x[j] + (k - 1) * 0.26, n + 1.2, str(n), ha="center", fontsize=12)
for j, inst in enumerate(INSTITUTIONS):
    kept = stages.loc[inst, "Analysis corpus"] / stages.loc[inst, "In scope (scope rules)"]
    ax.text(x[j], stages.loc[inst].max() + 9, f"{kept:.0%} kept", ha="center", style="italic", color="#444444", fontsize=13)
ax.set_xticks(x, INSTITUTIONS, fontsize=13)
ax.set_ylabel("documents")
ax.set_ylim(0, 95)
ax.legend(handles=[Patch(color="#555555", alpha=a, label=s) for s, a in shade.items()], frameon=False, loc="upper right")
save(fig, "figure_corpus_attrition_final.png")
