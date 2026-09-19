"""Week 4: two ways of measuring what an institution emphasises.

The project's hypotheses are about *frames* — whether an institution presents AI
as an economic opportunity, a rights problem, a security question, and so on. Two
routes to that, differing in where the categories come from:

  Dictionary scoring (theory-driven). The seven frames of the project proposal are
  operationalised as term lists. Each paragraph is scored on every frame, and
  institutions are compared on frame prevalence. The categories are fixed in
  advance, so the measure answers exactly the question the hypotheses ask — and
  cannot discover anything outside its own lists.

  Fightin' Words (data-driven). Monroe, Colaresi and Quinn's method for identifying
  the vocabulary that distinguishes two groups of texts, with no categories imposed.
  It answers "what separates these institutions?" rather than "how much of frame X
  does each use?", and can surface distinctions the dictionary has no entry for.

Neither validates the other, but agreement between them is evidence that a finding
is not an artefact of one operationalisation. Disagreement is equally informative,
and the report treats it as such.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import represent

# ---------------------------------------------------------------------------
# 1. Frame dictionary
# ---------------------------------------------------------------------------
# Seven frames from the project proposal. Terms were chosen to be unambiguous
# markers of the frame rather than merely topical: "competitiveness" signals the
# economic frame, while "market" alone would match any internal-market boilerplate.
# Each entry is matched as a whole token (or token bigram) after tokenisation.

FRAMES: dict[str, set[str]] = {
    "economic_opportunity": {
        "competitiveness", "competitive", "growth", "investment", "investments",
        "innovation", "productivity", "economy", "economic", "industry", "industrial",
        "business", "businesses", "smes", "startups", "market", "markets", "trade",
        "jobs", "prosperity", "uptake", "adoption", "scale", "leadership",
    },
    "fundamental_rights": {
        "rights", "fundamental", "dignity", "privacy", "discrimination",
        "discriminatory", "bias", "biased", "equality", "freedom", "freedoms",
        "charter", "data protection", "gdpr", "autonomy", "consent", "redress",
        "democratic", "democracy", "rule of law", "justice", "ethical", "ethics",
    },
    "security_geopolitical": {
        "security", "defence", "defense", "military", "threat", "threats",
        "sovereignty", "strategic", "autonomy", "geopolitical", "resilience",
        "dependency", "dependencies", "cyber", "cybersecurity", "attack", "attacks",
        "critical infrastructure", "national security", "espionage", "warfare",
    },
    "labour_social": {
        "workers", "worker", "employment", "employees", "labour", "labor", "jobs",
        "skills", "training", "reskilling", "upskilling", "inequality", "social",
        "wages", "working conditions", "trade unions", "displacement", "automation",
        "workplace", "education",
    },
    "consumer_protection": {
        "consumers", "consumer", "protection", "safety", "safe", "harm", "harms",
        "risk", "risks", "liability", "damages", "complaint", "complaints",
        "vulnerable", "trust", "trustworthy", "safeguards", "remedies", "recall",
    },
    "technological_innovation": {
        "technology", "technologies", "technological", "research", "development",
        "digital", "algorithm", "algorithms", "model", "models", "data", "training",
        "computing", "infrastructure", "testing", "sandbox", "sandboxes",
        "experimentation", "deployment", "science", "scientific",
    },
    "regulatory_governance": {
        "compliance", "enforcement", "obligations", "requirements", "supervision",
        "supervisory", "oversight", "authority", "authorities", "notified",
        "conformity", "assessment", "standards", "standardisation", "certification",
        "penalties", "sanctions", "governance", "transparency", "accountability",
        "audit", "monitoring", "register", "documentation",
    },
}


def score_frames(texts: pd.Series, drop_procedural: bool = False) -> pd.DataFrame:
    """Frame scores per text: matched terms per 100 tokens, one column per frame."""
    rows = []
    for text in texts:
        tokens = represent.tokenise(text, drop_generic=True, drop_procedural=drop_procedural)
        bigrams = {" ".join(tokens[i:i + 2]) for i in range(len(tokens) - 1)}
        unigrams = set(tokens)
        counts = {}
        for frame, terms in FRAMES.items():
            hits = sum(tokens.count(t) for t in terms if " " not in t and t in unigrams)
            hits += sum(1 for t in terms if " " in t and t in bigrams)
            counts[frame] = 100 * hits / max(len(tokens), 1)
        counts["n_tokens"] = len(tokens)
        rows.append(counts)
    return pd.DataFrame(rows, index=texts.index)


# ---------------------------------------------------------------------------
# 2. Fightin' Words
# ---------------------------------------------------------------------------

def fightin_words(texts_a: pd.Series, texts_b: pd.Series, spec: str = "baseline",
                  prior_strength: float = 0.01) -> pd.DataFrame:
    """Log-odds ratios with an informative Dirichlet prior (Monroe et al. 2008).

    Raw frequency comparison fails twice: common words dominate because the groups
    differ in size, and rare words produce enormous ratios on almost no evidence.
    The method fixes both — rates rather than counts, and a prior built from the
    pooled corpus that shrinks a term toward "no difference" in proportion to how
    little is known about it. Dividing by the standard error yields a z-score, so
    the ranking reflects both the size of a difference and the confidence in it.
    """
    ngrams, generic, procedural, rhetorical, _, min_df = represent.SPECIFICATIONS[spec]
    from sklearn.feature_extraction.text import CountVectorizer
    analyzer = lambda s: represent._ngrams(
        represent.tokenise(s, drop_generic=generic, drop_procedural=procedural,
                           drop_rhetorical=rhetorical), ngrams)
    vec = CountVectorizer(analyzer=analyzer, min_df=min_df)
    pooled = pd.concat([texts_a, texts_b])
    vec.fit(pooled)
    vocab = vec.get_feature_names_out()

    counts_a = np.asarray(vec.transform(texts_a).sum(axis=0)).ravel().astype(float)
    counts_b = np.asarray(vec.transform(texts_b).sum(axis=0)).ravel().astype(float)

    # Informative prior: the pooled rate, scaled by prior_strength.
    pooled_counts = counts_a + counts_b
    alpha = prior_strength * pooled_counts.sum() * (pooled_counts / pooled_counts.sum())

    a_tot, b_tot, alpha_tot = counts_a.sum(), counts_b.sum(), alpha.sum()
    odds_a = (counts_a + alpha) / (a_tot + alpha_tot - counts_a - alpha)
    odds_b = (counts_b + alpha) / (b_tot + alpha_tot - counts_b - alpha)
    delta = np.log(odds_a) - np.log(odds_b)
    variance = 1.0 / (counts_a + alpha) + 1.0 / (counts_b + alpha)
    z = delta / np.sqrt(variance)

    return (pd.DataFrame({"term": vocab, "z": z, "count_a": counts_a, "count_b": counts_b})
              .sort_values("z", ascending=False)
              .reset_index(drop=True))


def top_terms(fw: pd.DataFrame, n: int = 15) -> tuple[list[str], list[str]]:
    """The n most distinctive terms for group A and for group B."""
    return fw.head(n)["term"].tolist(), fw.tail(n)["term"].tolist()[::-1]
