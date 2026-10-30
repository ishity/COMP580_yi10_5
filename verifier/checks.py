"""Checks shared by the tool verifiers. Pure functions of their arguments and the files on disk."""

import json
import re
from pathlib import Path

from .rule import MAX_K, display, is_term, normalize, ranking_key, score

SELECTIONS = "selections"
RANKINGS = "rankings"


def result(valid, message, *, score=None, **details):
    return {
        "valid": valid,
        "score": float(valid) if score is None else score,
        "message": message,
        "details": details,
    }


def read_json(path):
    """The JSON in ``path``, or None when there is no such file."""
    path = Path(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def k_problem(k):
    if not 1 <= k <= MAX_K:
        return f"k must be from 1 to {MAX_K}, got {k}"
    return None


def years_problem(first_year, last_year):
    if first_year > last_year:
        return f"first_year {first_year} is after last_year {last_year}"
    return None


def paper_set_problem(dataset, paper_set_id):
    if not re.fullmatch(re.escape(dataset.value) + r"-c(0|[1-9][0-9]*)-[0-9]{4}-[0-9]{4}", paper_set_id):
        return f"paper_set_id {paper_set_id!r} is not a {dataset.value} selection, <dataset>-c<index>-<first>-<last>"
    return None


def rows_problem(rows, n, selected=None):
    """What is wrong with ranked (term, tf, df) rows over a scope of ``n`` papers, or None. ``selected`` is the
    selection's size when the scope is the selection itself, so that df <= tf and df <= selected."""
    terms = [term for term, _, _ in rows]
    if len(set(terms)) != len(terms):
        return "a term is ranked twice"
    for term, tf, df in rows:
        if not is_term(term) or normalize(term) != [term]:
            return f"{term!r} is not a normalized term"
        if tf < 1 or not 1 <= df <= n:
            return f"term {term!r} has impossible counts tf {tf}, df {df} for N {n}"
        if selected is not None and (df > tf or df > selected):
            return f"term {term!r} has df {df} over the selection, more than its tf {tf} or the {selected} papers"
    keys = [ranking_key(term, score(tf, df, n)) for term, tf, df in rows]
    if keys != sorted(keys):
        return "the terms are not ranked by score descending, then term ascending"
    return None


def shown(rows, n):
    """Ranked (term, tf, df) rows as (term, tf, df, displayed score)."""
    return [(term, tf, df, display(score(tf, df, n))) for term, tf, df in rows]
