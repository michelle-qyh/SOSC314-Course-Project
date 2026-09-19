"""Week 4: turning text into numbers — representation and preprocessing choices.

Week 3 deliberately stopped preprocessing at cleaned paragraphs, on the grounds
that tokenisation, stopword removal and weighting are model-specific rather than
properties of the corpus. This module makes those choices explicit and, more
importantly, *variable*: every analysis can be run under several representations so
that the effect of the choice is visible rather than assumed.

A representation is fixed by four decisions:

    tokens      how a string becomes a list of terms (lowercase, alphabetic,
                minimum length, optional EU-specific stopwords)
    ngrams      unigrams only, or unigrams plus bigrams
    weighting   raw counts or tf-idf
    pruning     minimum document frequency

The defaults below are the specification reported as the main analysis; the
variants are what the report's controlled comparison moves.
"""
from __future__ import annotations

import re

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Ordinary English function words. Kept separate from the domain list below so the
# two can be switched independently.
GENERIC_STOPWORDS = {
    "the", "of", "and", "to", "in", "a", "is", "that", "for", "on", "as", "with",
    "by", "be", "this", "are", "it", "or", "an", "which", "from", "at", "shall",
    "not", "have", "has", "been", "will", "their", "its", "such", "may", "can",
    "should", "would", "must", "these", "those", "any", "all", "other", "more",
    "also", "into", "under", "between", "where", "when", "than", "there", "they",
    "we", "our", "his", "her", "them", "he", "she", "do", "does", "no", "if",
}

# Terms that identify the *procedural furniture* of EU documents rather than their
# framing: the words every institution uses simply because it is an EU institution
# writing about a legislative file. Removing them is the sharpest preprocessing
# lever in this project, because leaving them in lets distinctive-language methods
# recover the institution from its own house style rather than from what it says.
EU_PROCEDURAL_STOPWORDS = {
    "european", "union", "eu", "commission", "parliament", "council", "member",
    "states", "state", "regulation", "directive", "article", "articles", "paragraph",
    "proposal", "proposals", "amendment", "amendments", "committee", "annex",
    "regulations", "directives", "decision", "decisions", "text", "act", "acts",
    "com", "doc", "document", "documents", "brussels", "official", "journal",
    "whereas", "thereof", "herein", "referred", "laying", "down", "amending",
    "pursuant", "accordance", "respect", "regard", "having", "adopted", "adopt",
    "rapporteur", "opinion", "report", "draft", "procedure", "reading", "session",
}

# The verbs by which EU institutions perform their own genre: Parliament resolutions
# are built from "calls on", "stresses", "welcomes"; Commission documents from
# "proposes" and "considers". These are the grammar of a document type, not a claim
# about AI, and they dominate distinctive-language results unless removed.
RHETORICAL_STOPWORDS = {
    "calls", "call", "stresses", "stress", "underlines", "underline", "notes",
    "note", "highlights", "highlight", "recalls", "recall", "welcomes", "welcome",
    "regrets", "regret", "considers", "consider", "emphasises", "emphasise",
    "reiterates", "urges", "urge", "invites", "invite", "requests", "request",
    "recommends", "recommend", "believes", "believe", "acknowledges", "points",
    "whereas", "having", "regard", "instructs", "takes", "expresses",
}

TOKEN = re.compile(r"[a-z]+")


def tokenise(text: str, min_length: int = 3, drop_generic: bool = True,
             drop_procedural: bool = False, drop_rhetorical: bool = False) -> list[str]:
    """Lowercase, keep alphabetic tokens, apply the selected stopword lists."""
    tokens = [t for t in TOKEN.findall(str(text).lower()) if len(t) >= min_length]
    if drop_generic:
        tokens = [t for t in tokens if t not in GENERIC_STOPWORDS]
    if drop_procedural:
        tokens = [t for t in tokens if t not in EU_PROCEDURAL_STOPWORDS]
    if drop_rhetorical:
        tokens = [t for t in tokens if t not in RHETORICAL_STOPWORDS]
    return tokens


SPECIFICATIONS = {
    # name               ngrams  generic procedural rhetorical weighting min_df
    "baseline":         ((1, 1), True,  False,     False,     "count",   5),
    "no_stopwords":     ((1, 1), False, False,     False,     "count",   5),
    "drop_procedural":  ((1, 1), True,  True,      False,     "count",   5),
    "drop_house_style": ((1, 1), True,  True,      True,      "count",   5),
    "with_bigrams":     ((1, 2), True,  True,      True,      "count",   5),
    "tfidf":            ((1, 1), True,  True,      True,      "tfidf",   5),
    "min_df_20":        ((1, 1), True,  True,      True,      "count",  20),
}


def build_matrix(texts: pd.Series, spec: str = "baseline"):
    """Return (matrix, vocabulary) for one named specification."""
    ngrams, generic, procedural, rhetorical, weighting, min_df = SPECIFICATIONS[spec]
    analyzer = lambda s: _ngrams(tokenise(s, drop_generic=generic, drop_procedural=procedural,
                                          drop_rhetorical=rhetorical), ngrams)
    cls = TfidfVectorizer if weighting == "tfidf" else CountVectorizer
    vec = cls(analyzer=analyzer, min_df=min_df)
    matrix = vec.fit_transform(texts)
    return matrix, vec.get_feature_names_out()


def _ngrams(tokens: list[str], ngram_range: tuple[int, int]) -> list[str]:
    low, high = ngram_range
    out = []
    for n in range(low, high + 1):
        if n == 1:
            out.extend(tokens)
        else:
            out.extend(" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    return out
