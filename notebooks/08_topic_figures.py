# %% [markdown]
# # 08 — Topic-model figures used in the slides and on the website
#
# Redraws the topic-model figures from the saved Week 5 outputs, so every number on
# the slides can be traced to a file in the repository:
#
#   figure_slide13_council_week4.png    Council's two largest Week 4 LDA topics (slide 13)
#   figure_slide14_highrisk_weighting.png  high-risk LDA topic: document- vs paragraph-weighted,
#                                        with 95% bootstrap intervals (slide 14)
#   figure_backupA1_nmf_highrisk.png    the five NMF topics that cover the high-risk rules (backup A1)
#   robust_leads_all_topics.csv         the non-overlap rule applied to every topic, not only
#                                        the three tracked ones (the "all 10 LDA topics" line on slide 14)
#
# Inputs (written by 05_topic_diagnostics.ipynb):
#   reports/week5_topic_diagnostics/bootstrap_topic_shares.csv
#   reports/week5_topic_diagnostics/weighting_comparison.csv
#
# Run from the repository root or from notebooks/:  python notebooks/08_topic_figures.py

# %%
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
IN_DIR = ROOT / "reports" / "week5_topic_diagnostics"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLOURS = {"Commission": "#1f4e79", "Parliament": "#c0504d", "Council": "#7f7f7f"}
INSTITUTIONS = ["Commission", "Parliament", "Council"]
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"], "font.size": 14,
                     "axes.spines.top": False, "axes.spines.right": False})

boot = pd.read_csv(IN_DIR / "bootstrap_topic_shares.csv")      # shares are proportions (0-1)
weights = pd.read_csv(IN_DIR / "weighting_comparison.csv")


def share(model, topic, institution):
    row = boot[(boot.model == model) & (boot.topic == topic) & (boot.institution == institution)].iloc[0]
    return 100 * row.point, 100 * row.ci_low, 100 * row.ci_high


# %% [markdown]
# ## 1. Robust leads across all topics
# Same rule as 05_topic_diagnostics.ipynb: a lead counts only if the leader's 95% interval
# does not overlap the runner-up's.

# %%
rows = []
for (model, topic), sub in boot.groupby(["model", "topic"]):
    sub = sub.sort_values("point", ascending=False)
    lead, second = sub.iloc[0], sub.iloc[1]
    rows.append(dict(model=model, topic=topic,
                     leader=lead.institution, leader_point=lead.point, leader_ci_low=lead.ci_low,
                     runner_up=second.institution, runner_up_point=second.point,
                     runner_up_ci_high=second.ci_high,
                     robust=bool(lead.ci_low > second.ci_high)))
all_leads = pd.DataFrame(rows)
all_leads.to_csv(IN_DIR / "robust_leads_all_topics.csv", index=False)
print(all_leads.to_string(index=False))
print("\nrobust leads:", all_leads[all_leads.robust][["model", "topic", "leader"]].to_dict("records"))

# %% [markdown]
# ## 2. Slide 13 — Council's two largest Week 4 topics
# Week 4 values (paragraph-weighted, Week 4 preprocessing) as reported in the Week 4 report
# (p. 3) and Figure 2a of 04_topic_modeling.ipynb. The Week 4 model is not refitted here.

# %%
fig, ax = plt.subplots(figsize=(8, 3.2))
labels = ["High-risk requirements &\nconformity assessment", "National authorities &\ncompliance enforcement"]
values = [24.9, 29.6]
ax.barh(labels, values, color=COLOURS["Council"], height=0.55)
for y, v in enumerate(values):
    ax.text(v + 0.5, y, f"{v}%", va="center")
ax.set_xlim(0, 40)
ax.set_xlabel("Average topic share of the Council's text (%)")
fig.suptitle("Initial LDA (Week 4): Council's two largest topics = 54.5%", x=0.02, ha="left", fontsize=15)
fig.tight_layout()
fig.savefig(FIG_DIR / "figure_slide13_council_week4.png", dpi=220, facecolor="white")
plt.close(fig)

# %% [markdown]
# ## 3. Slide 14 — high-risk LDA topic (T7): document- vs paragraph-weighted

# %%
HIGH_RISK = 7
order = ["Council", "Parliament", "Commission"]
fig, ax = plt.subplots(figsize=(10, 4.6))
for k, inst in enumerate(order):
    p, lo, hi = share("LDA", HIGH_RISK, inst)
    pw = 100 * weights[(weights.model == "LDA") & (weights.topic == HIGH_RISK)
                       & (weights.institution == inst)].paragraph_weighted.iloc[0]
    c = COLOURS[inst]
    ax.errorbar(p, k, xerr=[[p - lo], [hi - p]], fmt="o", color=c, ms=11, capsize=6, lw=2.5, zorder=3)
    ax.plot(pw, k, "o", mfc="white", mec=c, ms=11, mew=2, zorder=4)
    if pw - p > 3:
        ax.annotate("", xy=(p + 0.6, k + 0.22), xytext=(pw - 0.6, k + 0.22),
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.2, ls="--"))
    ax.text(49, k, f"{p:.1f}%  [{lo:.1f}, {hi:.1f}]", color=c, va="center")
    ax.text(49, k + 0.3, f"paragraph-weighted {pw:.1f}%", color=c, va="center", fontsize=12, style="italic")
ax.set_yticks(range(3), order)
for lab in ax.get_yticklabels():
    lab.set_color(COLOURS[lab.get_text()])
ax.invert_yaxis()
ax.set_xlim(0, 47)
ax.set_ylim(2.6, -0.5)
ax.set_xlabel("High-risk topic share of institution (%)")
ax.grid(axis="x", color="#e6e6e6")
ax.set_axisbelow(True)
handles = [Line2D([], [], marker="o", color="black", ms=10, lw=2, label="Document-weighted, 95% bootstrap CI (final)"),
           Line2D([], [], marker="o", mfc="white", mec="black", ms=10, mew=2, lw=0, label="Paragraph-weighted (for comparison)")]
fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.subplots_adjust(right=0.72)
fig.savefig(FIG_DIR / "figure_slide14_highrisk_weighting.png", dpi=300, facecolor="white")
plt.close(fig)

# %% [markdown]
# ## 4. Backup A1 — the five NMF topics that cover the high-risk rules

# %%
pieces = [(1, "High-risk systems\n(risk, high risk, providers)"),
          (4, "Authorities & enforcement\n(competent authorities, surveillance)"),
          (9, "Conformity assessment\n(notified bodies)"),
          (7, "Testing & harmonised law\n(real-world testing)"),
          (6, "Legal requirements\n(\"set out in Chapter III\")")]
offset = {"Commission": -0.25, "Parliament": 0.0, "Council": 0.25}
robust_nmf = all_leads[(all_leads.model == "NMF")].set_index("topic").robust
fig, ax = plt.subplots(figsize=(9.6, 5.0))
for k, (topic, label) in enumerate(pieces):
    for inst in INSTITUTIONS:
        p, lo, hi = share("NMF", topic, inst)
        ax.errorbar(p, k + offset[inst], xerr=[[p - lo], [hi - p]], fmt="o", color=COLOURS[inst],
                    ms=8, capsize=4, lw=2, label=inst if k == 0 else None)
    clear = bool(robust_nmf[topic])
    ax.text(19.4, k, "clear lead" if clear else "overlap", va="center", fontsize=13,
            color="#222222" if clear else "#888888", fontweight="bold" if clear else "normal")
ax.set_yticks(range(len(pieces)), [p[1] for p in pieces], fontsize=13)
ax.invert_yaxis()
ax.set_xlim(0, 19)
ax.set_xticks(range(0, 20, 5))
ax.set_xlabel("Topic share (%), equal weight per document, 95% interval", fontsize=13)
ax.grid(axis="x", color="#e6e6e6")
ax.set_axisbelow(True)
fig.legend(loc="upper center", ncol=3, frameon=False, fontsize=13)
fig.tight_layout(rect=(0, 0, 0.93, 0.93))
fig.savefig(FIG_DIR / "figure_backupA1_nmf_highrisk.png", dpi=250, facecolor="white")
plt.close(fig)

print("\nfigures written to", FIG_DIR)