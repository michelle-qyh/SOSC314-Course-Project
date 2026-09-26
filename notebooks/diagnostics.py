"""Week 5: diagnostics for the frame dictionary and the distinctive-language measure.

Week 4 reported that the dictionary was stable under six representation
specifications while Fightin' Words lost a fifth to a third of its top terms. That
comparison was incomplete in two ways, and this module tests both.

First, the specifications varied *preprocessing* — which words are removed before
counting — and the dictionary is structurally immune to that, because it only ever
counts terms on its own seven lists. Its apparent robustness was therefore partly a
restatement of what it ignores. The dimension that can move it is the content of the
lists themselves, which Week 4 never varied. `leave_one_term_out` and
`tightened_profile` test that dimension.

Second, the Week 4 specifications were all measured against the baseline, so the
three that vary counting (bigrams, tf-idf, min_df) also carried the stopword change.
`sensitivity_clean` separates them.

`bootstrap_profile` addresses a different question: not whether the measure is stable
under our choices, but whether the corpus is large enough to support the differences
we read off it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import measures
import represent

FRAMES = list(measures.FRAMES)

# Terms that are generic EU digital-policy vocabulary rather than markers of the
# frame they sit in. Identified by leave-one-term-out: each of these, removed alone,
# moves at least one frame share by more than two percentage points.
GENERIC_FRAME_TERMS: dict[str, set[str]] = {
    "technological_innovation": {"digital", "data", "model", "models", "development"},
    "economic_opportunity": {"market", "markets", "adoption", "scale"},
    "consumer_protection": {"risk", "risks"},
}


# ---------------------------------------------------------------------------
# Shared scoring machinery
# ---------------------------------------------------------------------------

def term_hits(paragraphs: pd.DataFrame) -> tuple[dict, np.ndarray]:
    """Per-paragraph hit counts for every dictionary term, plus token lengths.

    Computed once and reused: the leave-one-term-out diagnostic refits the profile
    152 times, and re-tokenising 27,732 paragraphs each time would be wasteful.
    """
    tokens = [represent.tokenise(t) for t in paragraphs["text"]]
    lengths = np.array([len(t) for t in tokens], dtype=float)
    unigrams = [set(t) for t in tokens]
    bigrams = [{" ".join(t[i:i + 2]) for i in range(len(t) - 1)} for t in tokens]

    hits: dict[str, dict[str, np.ndarray]] = {}
    for frame, terms in measures.FRAMES.items():
        hits[frame] = {}
        for term in terms:
            if " " in term:
                hits[frame][term] = np.array([1 if term in b else 0 for b in bigrams])
            else:
                hits[frame][term] = np.array(
                    [t.count(term) if term in u else 0 for t, u in zip(tokens, unigrams)])
    return hits, lengths


def profile(paragraphs: pd.DataFrame, hits: dict, lengths: np.ndarray,
            exclude: set[tuple[str, str]] | None = None) -> pd.DataFrame:
    """Frame profile as shares, optionally excluding (frame, term) pairs."""
    exclude = exclude or set()
    columns = {}
    for frame in FRAMES:
        total = np.zeros(len(paragraphs))
        for term, counts in hits[frame].items():
            if (frame, term) in exclude:
                continue
            total = total + counts
        columns[frame] = 100 * total / np.maximum(lengths, 1)

    scores = pd.DataFrame(columns)
    scores["doc_id"] = paragraphs["doc_id"].values
    scores["institution"] = paragraphs["institution"].values
    per_doc = scores.groupby(["doc_id", "institution"])[FRAMES].mean().reset_index()
    per_inst = per_doc.groupby("institution")[FRAMES].mean()
    return per_inst.T / per_inst.T.sum() * 100


# ---------------------------------------------------------------------------
# 1. Term influence
# ---------------------------------------------------------------------------

def leave_one_term_out(paragraphs: pd.DataFrame, hits: dict,
                       lengths: np.ndarray) -> pd.DataFrame:
    """Refit the profile without each term in turn; report the largest shift.

    A measure that depends on one word is not measuring a frame. This is the
    dictionary's equivalent of the term-turnover check applied to Fightin' Words.
    """
    base = profile(paragraphs, hits, lengths)
    rows = []
    for frame in FRAMES:
        for term in hits[frame]:
            alt = profile(paragraphs, hits, lengths, exclude={(frame, term)})
            rows.append({"frame": frame, "term": term,
                         "max_shift_pp": round(float((alt - base).abs().max().max()), 2)})
    return pd.DataFrame(rows).sort_values("max_shift_pp", ascending=False).reset_index(drop=True)


def tightened_profile(paragraphs: pd.DataFrame, hits: dict,
                      lengths: np.ndarray) -> pd.DataFrame:
    """The profile with GENERIC_FRAME_TERMS removed from their frames."""
    exclude = {(frame, term) for frame, terms in GENERIC_FRAME_TERMS.items() for term in terms}
    return profile(paragraphs, hits, lengths, exclude=exclude)


def frame_overlap() -> pd.DataFrame:
    """Terms appearing in more than one frame list."""
    rows = []
    for i, a in enumerate(FRAMES):
        for b in FRAMES[i + 1:]:
            shared = sorted(set(measures.FRAMES[a]) & set(measures.FRAMES[b]))
            if shared:
                rows.append({"frame_a": a, "frame_b": b, "shared_terms": ", ".join(shared)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Sampling
# ---------------------------------------------------------------------------

def bootstrap_profile(paragraphs: pd.DataFrame, hits: dict, lengths: np.ndarray,
                      n_boot: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Resample documents with replacement within institution; 95% percentile CIs.

    The corpus is the population of AI-related documents, not a sample from one, so
    the interval is not a sampling-error estimate in the survey sense. It answers a
    narrower question: how much of the observed difference depends on which
    particular documents an institution happened to produce? An institution holding
    28 documents cannot support a precise estimate however the text is processed.
    """
    columns = {}
    for frame in FRAMES:
        total = np.zeros(len(paragraphs))
        for counts in hits[frame].values():
            total = total + counts
        columns[frame] = 100 * total / np.maximum(lengths, 1)
    scores = pd.DataFrame(columns)
    scores["doc_id"] = paragraphs["doc_id"].values
    scores["institution"] = paragraphs["institution"].values
    per_doc = scores.groupby(["doc_id", "institution"])[FRAMES].mean().reset_index()

    def shares(docs: pd.DataFrame) -> pd.DataFrame:
        per_inst = docs.groupby("institution")[FRAMES].mean()
        return per_inst.T / per_inst.T.sum() * 100

    point = shares(per_doc)
    rng = np.random.default_rng(seed)
    by_inst = {name: group.reset_index(drop=True)
               for name, group in per_doc.groupby("institution")}
    draws = {inst: {frame: [] for frame in FRAMES} for inst in by_inst}
    for _ in range(n_boot):
        resampled = pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for g in by_inst.values()])
        drawn = shares(resampled)
        for inst in drawn.columns:
            for frame in FRAMES:
                draws[inst][frame].append(drawn.loc[frame, inst])

    rows = []
    for inst in point.columns:
        for frame in FRAMES:
            values = np.array(draws[inst][frame])
            rows.append({"institution": inst, "frame": frame,
                         "point": round(point.loc[frame, inst], 1),
                         "ci_low": round(float(np.percentile(values, 2.5)), 1),
                         "ci_high": round(float(np.percentile(values, 97.5)), 1)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Preprocessing sensitivity, disentangled
# ---------------------------------------------------------------------------

PAIRS = [("Commission", "Parliament"), ("Parliament", "Council"), ("Commission", "Council")]


def sensitivity_clean(paragraphs: pd.DataFrame, n_terms: int = 20) -> pd.DataFrame:
    """Top-term turnover, measuring each change against the right reference.

    Week 4 measured every specification against the baseline, so the three that vary
    counting also carried the stopword change and appeared as costly as it was. Here
    the stopword change is measured against the baseline, and the counting changes
    against the stopword-cleaned text, so each figure reflects one decision.
    """
    tops = {}
    for spec in ["baseline", "drop_house_style", "with_bigrams", "tfidf", "min_df_20"]:
        for a, b in PAIRS:
            fw = measures.fightin_words(paragraphs.loc[paragraphs.institution == a, "text"],
                                        paragraphs.loc[paragraphs.institution == b, "text"],
                                        spec=spec)
            top_a, top_b = measures.top_terms(fw, n_terms)
            tops[(spec, a, b)] = set(top_a)
            tops[(spec, b, a)] = set(top_b)

    def overlap(reference: str, spec: str) -> float:
        values = [len(tops[(reference, x, y)] & tops[(spec, x, y)]) / n_terms
                  for a, b in PAIRS for x, y in [(a, b), (b, a)]]
        return round(sum(values) / len(values), 2)

    rows = [{"change": "stopword lists", "reference": "baseline",
             "overlap": overlap("baseline", "drop_house_style")}]
    for spec in ["with_bigrams", "tfidf", "min_df_20"]:
        rows.append({"change": spec, "reference": "drop_house_style",
                     "overlap": overlap("drop_house_style", spec)})
    return pd.DataFrame(rows)
